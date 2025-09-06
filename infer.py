# Assumption: source file is "src.txt" with one space-separated sequence of token ids per line (no BOS/EOS)
#NE
import argparse, os
import torch, torch.nn.functional as F
from typing import List
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import pad_msk

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_src(path: str) -> List[List[int]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                out.append([])
            else:
                out.append(list(map(int, line.split())))
    return out

def batch_pad_src(src_batch: List[List[int]]):
    B = len(src_batch)
    S = max((len(s) for s in src_batch), default=0)
    src_tensor = torch.full((B, S), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(src_batch):
        if s:
            src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return src_tensor

def build_decoder_masks(tgt_tokens: torch.Tensor, pad_idx: int):
    B, T = tgt_tokens.shape
    tri = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt_tokens.device))
    fut = (~tri).unsqueeze(0).unsqueeze(0)
    dec_pad = pad_msk(tgt_tokens, pad_idx=pad_idx)
    return fut.to(tgt_tokens.device), dec_pad.to(tgt_tokens.device)

def greedy_generate(enc, dec, src_tensor: torch.Tensor, max_len: int):
    enc.eval(); dec.eval()
    src_tensor = src_tensor.to(DEVICE)
    with torch.no_grad():
        enc_pad = pad_msk(src_tensor, pad_idx=PAD_IDX).to(DEVICE)
        memory, _ = enc(src_tensor, enc_pad)
    B = src_tensor.size(0)
    cur = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    outputs = []
    for _ in range(max_len):
        self_attn_mask, self_key_padding_mask = build_decoder_masks(cur, PAD_IDX)
        cross_key_padding_mask = enc_pad
        with torch.no_grad():
            logits, _ = dec(cur, memory, self_attn_mask, self_key_padding_mask, cross_key_padding_mask)
        last_logits = logits[:, -1, :]
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)
        outputs.append(next_tokens)
        cur = torch.cat([cur, next_tokens], dim=1)
        just_finished = next_tokens.squeeze(1) == EOS_IDX
        finished = finished | just_finished
        if finished.all():
            break
    gen = torch.cat(outputs, dim=1)
    return gen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/src.txt")
    p.add_argument("--ckpt", default="checkpoint_epoch1.pt")
    p.add_argument("--vocab", type=int, default=5000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128*4)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    enc = TransformerEncoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, pos_encoding="rope").to(DEVICE)
    dec = TransformerDecoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, self_pos_encoding="rope").to(DEVICE)

    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=DEVICE)
        if "encoder" in ck: enc.load_state_dict(ck["encoder"])
        if "decoder" in ck: dec.load_state_dict(ck["decoder"])

    src_list = read_src(args.src)
    # batch and infer
    outputs_all = []
    for i in range(0, len(src_list), args.batch):
        batch = src_list[i:i+args.batch]
        src_tensor = batch_pad_src(batch).to(DEVICE)
        gen = greedy_generate(enc, dec, src_tensor, args.max_len)  # [B, gen_len]
        gen = gen.cpu().tolist()
        for seq in gen:
            # print token ids space separated
            print(" ".join(map(str, seq)))
            outputs_all.append(seq)

if __name__ == "__main__":
    main()
