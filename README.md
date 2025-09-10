# TransformerANLP Usage Instructions

This README provides instructions on how to run the training script for the TransformerANLP project.

## Prerequisites
- Ensure you have Python installed (version 3.8 or higher recommended).
- Install required dependencies (e.g., PyTorch, NumPy, etc.). You can install them using:
  ```bash
  pip install -r requirements.txt
  ```
- Prepare your dataset in the `/content/data` directory or update the `--data_dir` path accordingly.

## Running the Training Script
To train the Transformer model, use the following command. Adjust parameters as needed for your setup.

```bash
OUTDIR="/path/to/your/output/directory"  # Set your output directory
mkdir -p "$OUTDIR"                      # Create the output directory if it doesn't exist

echo "Starting training at $(date)"
python -u train.py \
  --data_dir /content/data \            # Path to your dataset
  --vocab 16003 \                       # Vocabulary size
  --d_model 256 \                       # Model dimension
  --num_heads 8 \                       # Number of attention heads
  --num_layers 4 \                      # Number of transformer layers
  --d_ff 1024 \                         # Feed-forward dimension
  --batch 32 \                          # Batch size
  --epochs 5 \                          # Number of epochs
  --pos_encoding rope \                  # Positional encoding type
  --out_dir "$OUTDIR" \                 # Output directory for logs and checkpoints
  --split \                             # Enable train/validation split
  |& tee "${OUTDIR}/training.log"       # Log output to file and console

echo "Finished training at $(date)"
echo "Last 20 lines of training.log:"
tail -n 20 "${OUTDIR}/training.log"
```

## Notes
- Replace `/path/to/your/output/directory` with your desired output path.
- Ensure the `--data_dir` points to your dataset directory.
- The script logs training progress to `training.log` in the specified `OUTDIR`.
- Modify hyperparameters (e.g., `--d_model`, `--epochs`) based on your requirements.

For additional details, refer to the project documentation or source files.
