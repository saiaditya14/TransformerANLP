# test.py
# Usage example:
# python test.py --ckpt exp/checkpoint_ep5_rope.pt --data_dir data --vocab 16003 --decoding beam --beam 5 --out preds.ids
import argparse, os, json
import torch, math
import sacrebleu
from typing import List
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import pad_msk

PAD_IDX=0; BOS_IDX=1; EOS_IDX=2

def read_id_file(path: str) -> List[List[int]]:
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if ln=="":
                out.append([])
            else:
                out.append(list(map(int, ln.split())))
    return out

def batch_pad_src(batch: List[List[int]]):
    B = len(batch); S = max((len(s) for s in batch), default=0)
    src = torch.full((B,S), PAD_IDX, dtype=torch.long)
    for i,s in enumerate(batch):
        if s: src[i,:len(s)] = torch.tensor(s, dtype=torch.long)
    return src

def build_decoder_masks(tgt_tokens: torch.Tensor):
    B,T = tgt_tokens.shape
    tri = torch.tril(torch.ones((T,T), dtype=torch.bool, device=tgt_tokens.device))
    fut = (~tri).unsqueeze(0).unsqueeze(0)
    dec_pad = pad_msk(tgt_tokens, pad_idx=PAD_IDX)
    return fut.to(tgt_tokens.device), dec_pad.to(tgt_tokens.device)

def greedy_step(enc, dec, src_tensor, max_len, device):
    enc.eval(); dec.eval()
    src_tensor = src_tensor.to(device)
    with torch.no_grad():
        enc_pad = pad_msk(src_tensor, pad_idx=PAD_IDX).to(device)
        memory,_ = enc(src_tensor, enc_pad)
    B = src_tensor.size(0)
    cur = torch.full((B,1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    outputs=[]
    for _ in range(max_len):
        fut, dec_pad = build_decoder_masks(cur)
        with torch.no_grad():
            logits,_ = dec(cur, memory, fut, dec_pad, enc_pad)
        last = logits[:, -1, :]
        next_tokens = last.argmax(dim=-1, keepdim=True)
        outputs.append(next_tokens)
        cur = torch.cat([cur, next_tokens], dim=1)
        finished = finished | (next_tokens.squeeze(1) == EOS_IDX)
        if finished.all(): break
    return torch.cat(outputs, dim=1).cpu().tolist()

def topk_step(enc, dec, src_tensor, max_len, top_k, device):
    enc.eval(); dec.eval()
    src_tensor = src_tensor.to(device)
    with torch.no_grad():
        enc_pad = pad_msk(src_tensor, pad_idx=PAD_IDX).to(device)
        memory,_ = enc(src_tensor, enc_pad)
    B = src_tensor.size(0)
    cur = torch.full((B,1), BOS_IDX, dtype=torch.long, device=device)
    outputs=[]
    for _ in range(max_len):
        fut, dec_pad = build_decoder_masks(cur)
        with torch.no_grad():
            logits,_ = dec(cur, memory, fut, dec_pad, enc_pad)
        last = logits[:,-1,:]
        probs = torch.softmax(last, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        next = torch.multinomial(topk_probs, num_samples=1)
        next_tokens = topk_idx.gather(-1, next)
        outputs.append(next_tokens)
        cur = torch.cat([cur, next_tokens], dim=1)
    return torch.cat(outputs, dim=1).cpu().tolist()

def beam_search(enc, dec, src_tensor, max_len, beam_size, device):
    enc.eval(); dec.eval()
    src_tensor = src_tensor.to(device)
    with torch.no_grad():
        enc_pad = pad_msk(src_tensor, pad_idx=PAD_IDX).to(device)
        memory,_ = enc(src_tensor, enc_pad)
    B = src_tensor.size(0)
    all_beams = []
    for i in range(B):
        mem_i = memory[i:i+1]  # [1,S,D]
        enc_pad_i = enc_pad[i:i+1]
        beams = [([BOS_IDX], 0.0, False)]  # (tokens, logprob, finished)
        for _ in range(max_len):
            next_beams=[]
            for tokens, score, finished in beams:
                if finished:
                    next_beams.append((tokens, score, True)); continue
                cur = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1,L]
                fut, dec_pad = build_decoder_masks(cur)
                with torch.no_grad():
                    logits,_ = dec(cur, mem_i, fut, dec_pad, enc_pad_i)
                logp = torch.log_softmax(logits[:,-1,:], dim=-1).squeeze(0)  # [V]
                topk_logp, topk_idx = torch.topk(logp, k=beam_size)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    ntoks = tokens + [int(idx)]
                    nscore = score + float(lp)
                    nfinished = (idx == EOS_IDX)
                    next_beams.append((ntoks, nscore, nfinished))
            # prune
            next_beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = next_beams
            # stop if all beams finished
            if all(b[2] for b in beams): break
        # choose best non-empty beam (remove BOS)
        best = max(beams, key=lambda x: x[1])[0][1:]  # drop BOS
        all_beams.append(best)
    return all_beams

def compute_bleu(preds: List[str], refs: List[str]):
    # sacrebleu expects list of hypothesis strings and list of reference strings
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--decoding", choices=("greedy","beam","topk"), default="greedy")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--out", default="preds.ids")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    enc_args = ck.get("args", {})
    pos_enc = enc_args.get("pos_encoding", "rope")
    enc = TransformerEncoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, pos_encoding=pos_enc).to(device)
    dec = TransformerDecoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, self_pos_encoding=pos_enc).to(device)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])

    src_test = os.path.join(args.data_dir,"test.src.txt")
    tgt_test = os.path.join(args.data_dir,"test.tgt.txt")
    if not os.path.exists(src_test):
        raise SystemExit("test split not found")
    src_lines = read_id_file(src_test)
    tgt_lines = read_id_file(tgt_test)

    # perform batched decoding in batches of size BATCH
    BATCH = 32
    outs = []
    for i in range(0, len(src_lines), BATCH):
        batch = src_lines[i:i+BATCH]
        src_tensor = batch_pad_src(batch)
        if args.decoding == "greedy":
            out_ids = greedy_step(enc, dec, src_tensor, args.max_len, device)
        elif args.decoding == "topk":
            out_ids = topk_step(enc, dec, src_tensor, args.max_len, args.topk, device)
        else:
            out_ids = beam_search(enc, dec, src_tensor, args.max_len, args.beam, device)
        outs.extend(out_ids)

    # write id outputs
    with open(args.out, "w", encoding='utf-8') as f:
        for seq in outs:
            f.write(" ".join(map(str, seq)) + "\n")

    # detokenize and compute BLEU using user's detokenize.py approach — here we'll detokenize using SentencePiece if model exists
    # For simplicity, compare ids directly by stripping special tokens and comparing numeric strings via sacrebleu (user should detokenize for real BLEU)
    # convert reference and preds to space-joined token strings (token ids without special reserved ids)
    def ids_to_str(seq):
        return " ".join(str(x) for x in seq if x not in (PAD_IDX,BOS_IDX,EOS_IDX))
    preds = [ids_to_str(s) for s in outs]
    refs = [ids_to_str(s) for s in tgt_lines]
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    results = {"decoding": args.decoding, "bleu": bleu.score, "n_preds": len(preds)}
    with open("results.json","w") as jf:
        json.dump(results, jf, indent=2)
    print("BLEU:", bleu.score)
    print("Wrote preds ids to", args.out)
if __name__ == "__main__":
    main()
