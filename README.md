v4∙LatestCopyPublishTransformer Model Training
How to Run
Set output directory and run training:
bashOUTDIR="/path/to/your/output/directory"
mkdir -p "$OUTDIR"

python -u train.py \
  --data_dir /path/to/data \
  --vocab 16003 \
  --d_model 256 \
  --num_heads 8 \
  --num_layers 4 \
  --d_ff 1024 \
  --batch 32 \
  --epochs 5 \
  --pos_encoding rpb \
  --out_dir "$OUTDIR" \
  --split \
  |& tee "${OUTDIR}/training.log"
This will train a transformer model and save logs to the output directory.
