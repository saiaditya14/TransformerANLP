Requirements

Python 3.8+

PyTorch installed (GPU recommended).

pip install torch

Data (expected)

Place token-id files in data/:

data/src.txt — one source sentence per line as space-separated token ids

data/tgt.txt — one target sentence per line as space-separated token ids
Special token indices used by scripts: PAD=0, BOS=1, EOS=2.

If you don’t have train/val/test splits, the trainer can create them from those two files with --split.

Train (example)

Run from terminal or Colab bash cell. Adjust args as needed.

OUTDIR="/content/drive/MyDrive/TransformerANLP_exps/rope"
mkdir -p "$OUTDIR"

python -u train.py \
  --data_dir /content/data \
  --vocab 16003 \
  --d_model 256 \
  --num_heads 8 \
  --num_layers 4 \
  --d_ff 1024 \
  --batch 32 \
  --epochs 10 \
  --pos_encoding rope \
  --out_dir "$OUTDIR" \
  --split \
  |& tee "${OUTDIR}/training.log"


Checkpoints saved to OUTDIR as checkpoint_ep{ep}_{pos}.pt (contain encoder+decoder+args).

Loss CSV: loss_{pos}.csv in OUTDIR.

training.log contains stdout/stderr.

Inference (basic)

Assuming infer.py exists and accepts a checkpoint and input file. Use --help to confirm flags.

Example (adjust to your infer.py flags):

python infer.py \
  --ckpt /content/drive/MyDrive/TransformerANLP_exps/rope/checkpoint_ep10_rope.pt \
  --input /content/data/test.src.txt \
  --output /content/drive/MyDrive/TransformerANLP_exps/rope/preds.txt \
  --vocab 16003 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 1024 \
  --pos_encoding rope


If infer.py outputs token ids, run detokenise.py to convert ids → readable text.

Quick Colab tips

Mount Drive before using /content/drive:

from google.colab import drive
drive.mount('/content/drive')


Verify GPU:

import torch
print(torch.cuda.is_available())


If re-running same OUTDIR and want to keep old logs, append: |& tee -a "${OUTDIR}/training.log"

Note about resuming

Current checkpoints include only model weights and args (no optimizer state or epoch). You can load the weights to continue training, but optimizer state (Adam moments) will be fresh. If you want exact resumes later, add optimizer+epoch to saved checkpoint.

If you want, I can paste a one-cell Colab snippet that loads a checkpoint and runs N more epochs (no edits to train.py) — otherwise you’re good to go.
