# train.py
# Assumption: data/{src.txt,tgt.txt} exist (token-id lines). If no splits, this script will create train/val/test splits.
import argparse, os, random, csv, json
from typing import Tuple, List
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import pad_msk

PAD_IDX = 0; BOS_IDX = 1; EOS_IDX = 2

class ParallelIdDataset(Dataset):
    def __init__(self, src_lines: List[List[int]], tgt_lines: List[List[int]]):
        assert len(src_lines) == len(tgt_lines)
        self.src = src_lines; self.tgt = tgt_lines
    def __len__(self): return len(self.src)
    def __getitem__(self, idx): return self.src[idx], self.tgt[idx]

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

def write_id_file(path: str, data: List[List[int]]):
    with open(path,'w',encoding='utf-8') as f:
        for seq in data:
            f.write(" ".join(map(str, seq)) + "\n")

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    B = len(srcs)
    S = max((len(s) for s in srcs), default=0)
    T = max((len(t) + 1 for t in tgts), default=1)
    src_tensor = torch.full((B,S), PAD_IDX, dtype=torch.long)
    tgt_out = torch.full((B,T), PAD_IDX, dtype=torch.long)
    tgt_in = torch.full((B,T), PAD_IDX, dtype=torch.long)
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        if s: src_tensor[i,:len(s)] = torch.tensor(s, dtype=torch.long)
        if len(t)==0 or t[-1]!=EOS_IDX: t2 = list(t)+[EOS_IDX]
        else: t2 = list(t)
        L = len(t2)
        tgt_out[i,:L] = torch.tensor(t2, dtype=torch.long)
        tgt_in[i,0] = BOS_IDX
        if L>1:
            tgt_in[i,1:L] = torch.tensor(t2[:-1], dtype=torch.long)
    return src_tensor, tgt_in, tgt_out

def make_splits(src_path, tgt_path, out_dir, val_frac=0.05, test_frac=0.05, seed=42):
    src = read_id_file(src_path); tgt = read_id_file(tgt_path)
    assert len(src)==len(tgt)
    N = len(src)
    idx = list(range(N))
    random.Random(seed).shuffle(idx)
    t_cnt = int(N * (1 - val_frac - test_frac))
    v_cnt = int(N * val_frac)
    train_idx = idx[:t_cnt]
    val_idx = idx[t_cnt:t_cnt+v_cnt]
    test_idx = idx[t_cnt+v_cnt:]
    def subset(idxs):
        return [src[i] for i in idxs], [tgt[i] for i in idxs]
    os.makedirs(out_dir, exist_ok=True)
    s_tr, t_tr = subset(train_idx)
    s_v, t_v = subset(val_idx)
    s_te, t_te = subset(test_idx)
    write_id_file(os.path.join(out_dir,'train.src.txt'), s_tr)
    write_id_file(os.path.join(out_dir,'train.tgt.txt'), t_tr)
    write_id_file(os.path.join(out_dir,'val.src.txt'), s_v)
    write_id_file(os.path.join(out_dir,'val.tgt.txt'), t_v)
    write_id_file(os.path.join(out_dir,'test.src.txt'), s_te)
    write_id_file(os.path.join(out_dir,'test.tgt.txt'), t_te)
    return

def build_masks(src, tgt_in, device):
    enc_pad = pad_msk(src, pad_idx=PAD_IDX).to(device)
    dec_pad = pad_msk(tgt_in, pad_idx=PAD_IDX).to(device)
    T = tgt_in.size(1)
    tri = torch.tril(torch.ones((T,T), dtype=torch.bool, device=device))
    fut = (~tri).unsqueeze(0).unsqueeze(0)
    return enc_pad, fut, dec_pad

def train_epoch(enc, dec, loader, opt, crit, device, args):
    enc.train(); dec.train()
    total_loss=0.0; steps=0
    for src, tgt_in, tgt_out in loader:
        src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)
        enc_pad, fut, dec_pad = build_masks(src, tgt_in, device)
        mem,_ = enc(src, enc_pad)
        logits,_ = dec(tgt_in, mem, fut, dec_pad, enc_pad)
        B,T,V = logits.shape
        loss = crit(logits.view(B*T,V), tgt_out.view(B*T))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 1.0)
        opt.step()
        total_loss += loss.item(); steps += 1
    return total_loss/steps if steps>0 else 0.0

def eval_epoch(enc, dec, loader, crit, device, args):
    enc.eval(); dec.eval()
    total_loss=0.0; steps=0
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)
            enc_pad, fut, dec_pad = build_masks(src, tgt_in, device)
            mem,_ = enc(src, enc_pad)
            logits,_ = dec(tgt_in, mem, fut, dec_pad, enc_pad)
            B,T,V = logits.shape
            loss = crit(logits.view(B*T,V), tgt_out.view(B*T))
            total_loss += loss.item(); steps += 1
    return total_loss/steps if steps>0 else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--vocab", type=int, default=5000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_encoding", choices=("rope","rpb"), default="rope")
    p.add_argument("--out_dir", default="exp")
    p.add_argument("--split", action="store_true", help="create train/val/test splits from data/src.txt,data/tgt.txt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.split:
        src_path = os.path.join(args.data_dir,"src.txt")
        tgt_path = os.path.join(args.data_dir,"tgt.txt")
        make_splits(src_path, tgt_path, args.data_dir)

    train_src = os.path.join(args.data_dir,"train.src.txt")
    train_tgt = os.path.join(args.data_dir,"train.tgt.txt")
    val_src = os.path.join(args.data_dir,"val.src.txt")
    val_tgt = os.path.join(args.data_dir,"val.tgt.txt")
    if not (os.path.exists(train_src) and os.path.exists(train_tgt)):
        raise SystemExit("Train split not found; run with --split or prepare train.src.txt/train.tgt.txt in data_dir")

    train_src_lines = read_id_file(train_src); train_tgt_lines = read_id_file(train_tgt)
    val_src_lines = read_id_file(val_src); val_tgt_lines = read_id_file(val_tgt)

    train_ds = ParallelIdDataset(train_src_lines, train_tgt_lines)
    val_ds = ParallelIdDataset(val_src_lines, val_tgt_lines)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=2)

    enc = TransformerEncoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, pos_encoding=args.pos_encoding).to(device)
    dec = TransformerDecoder(args.vocab, args.d_model, args.num_layers, args.num_heads, args.d_ff, pad_idx=PAD_IDX, self_pos_encoding=args.pos_encoding).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = optim.Adam(params, lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    csv_path = os.path.join(args.out_dir, f"loss_{args.pos_encoding}.csv")
    with open(csv_path, "w", newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch","train_loss","val_loss"])
        for ep in range(1, args.epochs+1):
            tr = train_epoch(enc, dec, train_loader, opt, crit, device, args)
            va = eval_epoch(enc, dec, val_loader, crit, device, args)
            writer.writerow([ep, tr, va])
            print(f"epoch {ep} train {tr:.4f} val {va:.4f}")
            ckpt = {"encoder": enc.state_dict(), "decoder": dec.state_dict(), "args": vars(args)}
            torch.save(ckpt, os.path.join(args.out_dir, f"checkpoint_ep{ep}_{args.pos_encoding}.pt"))
    # save final config
    with open(os.path.join(args.out_dir,"train_args.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)

if __name__ == "__main__":
    main()
