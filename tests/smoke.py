# smoke_test_with_ckpt.py
# Saves a tiny checkpoint and verifies load + inference; optionally calls test.py and infer.py if present.
import argparse, os, subprocess, sys
import torch, torch.nn as nn
from utils import pad_msk
from encoder import TransformerEncoder
from decoder import TransformerDecoder

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

def save_tiny_checkpoint(path, vocab=64, d_model=32, num_heads=4, num_layers=1, d_ff=64, pos_encoding="rope", device=torch.device("cpu")):
    enc = TransformerEncoder(vocab, d_model, num_layers, num_heads, d_ff, pad_idx=PAD_IDX, pos_encoding=pos_encoding).to(device)
    dec = TransformerDecoder(vocab, d_model, num_layers, num_heads, d_ff, pad_idx=PAD_IDX, self_pos_encoding=pos_encoding).to(device)
    ck = {"encoder": enc.state_dict(), "decoder": dec.state_dict(), "args": {"pos_encoding": pos_encoding}}
    torch.save(ck, path)
    return path

def load_and_run_inference(ckpt_path, device=torch.device("cpu")):
    ck = torch.load(ckpt_path, map_location=device)
    pos = ck.get("args", {}).get("pos_encoding", "rope")
    vocab = ck.get("args", {}).get("vocab", 64)
    # create fresh models with conservative defaults that match what we used to save above
    enc2 = TransformerEncoder(vocab if vocab else 64, 32, 1, 4, 64, pad_idx=PAD_IDX, pos_encoding=pos).to(device)
    dec2 = TransformerDecoder(vocab if vocab else 64, 32, 1, 4, 64, pad_idx=PAD_IDX, self_pos_encoding=pos).to(device)
    enc2.load_state_dict(ck["encoder"])
    dec2.load_state_dict(ck["decoder"])
    # tiny dummy input
    B, S = 2, 6
    src = torch.randint(3, 64, (B, S), dtype=torch.long, device=device)
    src[0, -2:] = PAD_IDX
    enc_pad = pad_msk(src, pad_idx=PAD_IDX).to(device)
    enc2.eval(); dec2.eval()
    with torch.no_grad():
        memory, _ = enc2(src, enc_pad)
    # greedy decode a short prefix
    cur = torch.full((B,1), BOS_IDX, dtype=torch.long, device=device)
    outs = []
    for _ in range(8):
        tri = torch.tril(torch.ones((cur.size(1), cur.size(1)), dtype=torch.bool, device=device))
        fut = (~tri).unsqueeze(0).unsqueeze(0).to(device)
        dec_pad = pad_msk(cur, pad_idx=PAD_IDX).to(device)
        with torch.no_grad():
            logits_ret = dec2(cur, memory, fut, dec_pad, enc_pad)
            logits = logits_ret[0] if (isinstance(logits_ret, tuple) and len(logits_ret) >= 1) else logits_ret
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        outs.append(next_tok)
        cur = torch.cat([cur, next_tok], dim=1)
    out_tensor = torch.cat(outs, dim=1)
    return out_tensor.cpu().tolist()

def try_run_script(script, args_list):
    if not os.path.exists(script):
        return False, f"{script} not found"
    cmd = [sys.executable, script] + args_list
    try:
        subprocess.check_call(cmd)
        return True, f"{script} ran successfully"
    except subprocess.CalledProcessError as e:
        return False, f"{script} failed (exit {e.returncode})"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt", default="tmp_ckpt.pt")
    p.add_argument("--attempt_scripts", action="store_true", help="Try to call test.py and infer.py as subprocesses if present")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("Saving tiny checkpoint to", args.ckpt)
    save_tiny_checkpoint(args.ckpt, device=device)

    print("Loading checkpoint and running a tiny inference with fresh models...")
    out = load_and_run_inference(args.ckpt, device=device)
    print("Inference output (id lists):", out)

    if args.attempt_scripts:
        print("Attempting to run test.py (greedy) with the tiny checkpoint...")
        success, msg = try_run_script("test.py", ["--ckpt", args.ckpt, "--data_dir", "data", "--vocab", "64", "--decoding", "greedy", "--out", "tmp_preds_from_test.ids"])
        print("test.py:", msg)
        print("Attempting to run infer.py with the tiny checkpoint...")
        success2, msg2 = try_run_script("infer.py", ["--ckpt", args.ckpt, "--src", "data/src.txt", "--vocab", "64", "--d_model", "32", "--num_heads", "4", "--num_layers", "1", "--d_ff", "64", "--max_len", "10", "--batch", "2"])
        print("infer.py:", msg2)

    print("Done. Remove", args.ckpt, "if you like.")

if __name__ == "__main__":
    main()
