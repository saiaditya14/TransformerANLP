# Assumption: dataset is two files data/src.txt and data/tgt.txt with token ids per line separated by spaces (e.g. "12 45 78 2")
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import pad_msk

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
VOCAB_MIN = 1000
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 2
D_FF = 4 * D_MODEL
BATCH = 32
LR = 1e-3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"

class ParallelTxtIdDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str):
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_lines = [list(map(int, line.strip().split())) for line in f if line.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_lines = [list(map(int, line.strip().split())) for line in f if line.strip()]
        assert len(self.src_lines) == len(self.tgt_lines)
    def __len__(self):
        return len(self.src_lines)
    def __getitem__(self, idx):
        return self.src_lines[idx], self.tgt_lines[idx]

def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    srcs, tgts = zip(*batch)
    B = len(srcs)
    S = max(len(s) for s in srcs)
    T = max(len(t) + 1 for t in tgts)
    src_tensor = torch.full((B, S), PAD_IDX, dtype=torch.long)
    tgt_output = torch.full((B, T), PAD_IDX, dtype=torch.long)
    tgt_input = torch.full((B, T), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        if len(t) == 0 or t[-1] != EOS_IDX:
            t_with_eos = list(t) + [EOS_IDX]
        else:
            t_with_eos = list(t)
        L = len(t_with_eos)
        tgt_output[i, :L] = torch.tensor(t_with_eos, dtype=torch.long)
        tgt_input[i, 0] = BOS_IDX
        if L > 1:
            tgt_input[i, 1:L] = torch.tensor(t_with_eos[:-1], dtype=torch.long)
    return src_tensor, tgt_input, tgt_output

src_path = os.path.join(DATA_DIR, "src.txt")
tgt_path = os.path.join(DATA_DIR, "tgt.txt")
dataset = ParallelTxtIdDataset(src_path, tgt_path)
max_id = 0
for s, t in dataset:
    if s: max_id = max(max_id, max(s))
    if t: max_id = max(max_id, max(t))
VOCAB = max(VOCAB_MIN, max_id + 1)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=2)

enc = TransformerEncoder(VOCAB, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, pad_idx=PAD_IDX, pos_encoding="rope").to(DEVICE)
dec = TransformerDecoder(VOCAB, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, pad_idx=PAD_IDX, self_pos_encoding="rope").to(DEVICE)
params = list(enc.parameters()) + list(dec.parameters())
opt = optim.Adam(params, lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch in range(1, EPOCHS + 1):
    enc.train(); dec.train()
    total_loss = 0.0
    for step, (src, decoder_input, decoder_target) in enumerate(loader, start=1):
        src = src.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_target = decoder_target.to(DEVICE)
        enc_pad = pad_msk(src, pad_idx=PAD_IDX).to(DEVICE)
        dec_pad = pad_msk(decoder_input, pad_idx=PAD_IDX).to(DEVICE)
        tri = torch.tril(torch.ones((decoder_input.size(1), decoder_input.size(1)), dtype=torch.bool, device=DEVICE))
        fut = (~tri).unsqueeze(0).unsqueeze(0)
        memory, _ = enc(src, enc_pad)
        logits, _ = dec(decoder_input, memory, fut, dec_pad, enc_pad)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), decoder_target.view(B * T))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        total_loss += loss.item()
        if step % 50 == 0:
            print(f"Epoch {epoch} step {step} avg_loss {total_loss / step:.4f}")
    print(f"Epoch {epoch} finished avg_loss {total_loss / len(loader):.4f}")
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict()}, f"checkpoint_epoch{epoch}.pt")
