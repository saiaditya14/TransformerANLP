# detokenize.py
# Usage examples:
#  python detokenize.py --mode spm --model data/spm.model --in preds.ids --out preds.txt
#  python detokenize.py --mode word --vocab data/vocab.txt --in preds.ids --out preds.txt
import argparse, os
import sentencepiece as spm

def load_spm(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def detok_spm_line(sp, id_line, shift=3):
    if id_line.strip() == "":
        return ""
    ids = [int(x) for x in id_line.strip().split()]
    # remove special reserved ids if present (PAD=0,BOS=1,EOS=2)
    ids = [i for i in ids if i not in (0,1,2)]
    # shift back
    ids = [i - shift for i in ids]
    # filter out negative/invalid just in case
    ids = [i for i in ids if i >= 0]
    if not ids:
        return ""
    return sp.DecodeIds(ids)

def load_word_vocab(vocab_path):
    # vocab file: one token per line; index 0 -> token id = 3 (because of shift)
    toks = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            toks.append(line.rstrip("\n"))
    return toks

def detok_word_line(vocab, id_line, shift=3):
    if id_line.strip() == "":
        return ""
    ids = [int(x) for x in id_line.strip().split()]
    ids = [i for i in ids if i not in (0,1,2)]
    toks = []
    for i in ids:
        j = i - shift
        if 0 <= j < len(vocab):
            toks.append(vocab[j])
        else:
            toks.append("<UNK>")
    return " ".join(toks)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("spm","word"), required=True)
    p.add_argument("--model", help="path to .model (for spm mode)")
    p.add_argument("--vocab", help="path to vocab.txt (for word mode)")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--shift", type=int, default=3, help="id shift used in preprocess (default 3)")
    args = p.parse_args()

    if args.mode == "spm":
        if not args.model or not os.path.exists(args.model):
            raise SystemExit("SPM model missing")
        sp = load_spm(args.model)
        detok_fn = lambda line: detok_spm_line(sp, line, shift=args.shift)
    else:
        if not args.vocab or not os.path.exists(args.vocab):
            raise SystemExit("vocab file missing")
        vocab = load_word_vocab(args.vocab)
        detok_fn = lambda line: detok_word_line(vocab, line, shift=args.shift)

    with open(args.infile, "r", encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        for ln in fin:
            s = detok_fn(ln)
            fout.write(s + "\n")

if __name__ == "__main__":
    main()
