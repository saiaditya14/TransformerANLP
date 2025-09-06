# preprocess.py
# Trains SentencePiece (joint by default) and writes id sequences compatible with train.py
import os, argparse, sentencepiece as spm

def train_sentencepiece(input_path, model_prefix, vocab_size, model_type="bpe", character_coverage=0.9995):
    spm.SentencePieceTrainer.Train(
        f"--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        f"--model_type={model_type} --character_coverage={character_coverage} --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1"
    )
    # Note: we set pad_id=0 and unk_id=1; we will remap ids by +3 below to reserve PAD/BOS/EOS slots

def encode_file(sp_processor, in_path, out_path, shift=3, append_eos=False):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            ids = sp_processor.EncodeAsIds(line) if line else []
            # remap ids: add shift (to reserve 0=PAD,1=BOS,2=EOS)
            ids = [str(i + shift) for i in ids]
            if append_eos:
                # append EOS token id = 2
                ids.append(str(2))
            fout.write(" ".join(ids) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)   # raw source text, one sentence per line
    p.add_argument("--tgt", required=True)   # raw target text, one sentence per line
    p.add_argument("--out_dir", default="data")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--joint", action="store_true", help="Train a joint spm model on src+tgt concatenated")
    p.add_argument("--model_type", default="bpe")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined = os.path.join(args.out_dir, "combined_for_spm.txt")
    if args.joint:
        # concatenate src + tgt for joint vocab
        with open(combined, "w", encoding="utf-8") as out:
            for path in (args.src, args.tgt):
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        out.write(ln)
        train_input = combined
        sp_prefix = os.path.join(args.out_dir, "spm_joint")
    else:
        # train separate models; we will train src and tgt separately
        train_input = None
        sp_prefix = os.path.join(args.out_dir, "spm_src")  # used for src; tgt will use spm_tgt

    if args.joint:
        train_sentencepiece(train_input, sp_prefix, args.vocab_size, model_type=args.model_type)
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_prefix + ".model")
        # encode src and tgt with same model
        src_out = os.path.join(args.out_dir, "src.txt")
        tgt_out = os.path.join(args.out_dir, "tgt.txt")
        encode_file(sp, args.src, src_out, shift=3, append_eos=False)
        encode_file(sp, args.tgt, tgt_out, shift=3, append_eos=True)
    else:
        # src model
        train_sentencepiece(args.src, os.path.join(args.out_dir, "spm_src"), args.vocab_size, model_type=args.model_type)
        sp_src = spm.SentencePieceProcessor(); sp_src.Load(os.path.join(args.out_dir, "spm_src.model"))
        train_sentencepiece(args.tgt, os.path.join(args.out_dir, "spm_tgt"), args.vocab_size, model_type=args.model_type)
        sp_tgt = spm.SentencePieceProcessor(); sp_tgt.Load(os.path.join(args.out_dir, "spm_tgt.model"))
        encode_file(sp_src, args.src, os.path.join(args.out_dir, "src.txt"), shift=3, append_eos=False)
        encode_file(sp_tgt, args.tgt, os.path.join(args.out_dir, "tgt.txt"), shift=3, append_eos=True)

    # write vocab info for reference (note: ids were shifted by +3; 0=PAD,1=BOS,2=EOS)
    with open(os.path.join(args.out_dir, "tokenizer_note.txt"), "w", encoding="utf-8") as f:
        f.write("SentencePiece model(s) saved in this folder (.model/.vocab)\n")
        f.write("IMPORTANT: token ids in src.txt/tgt.txt have been shifted by +3 to reserve:\n")
        f.write("  0 = PAD, 1 = BOS, 2 = EOS\n")
        f.write("You must construct your model with VOCAB >= (sp_vocab_size + 3)\n")

if __name__ == "__main__":
    main()
