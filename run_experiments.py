# run_experiments.py
# Runs two experiments (rope vs rpb), trains and plots train/val loss curves.
import subprocess, os, json, matplotlib.pyplot as plt

EXPERIMENTS = [
    {"name":"rope","pos_encoding":"rope"},
    {"name":"rpb","pos_encoding":"rpb"},
]

TRAIN_PY = "train.py"
DATA_DIR = "data"
OUT_DIR_BASE = "exp"
VOCAB = 16003  # sp_vocab + 3
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 2
D_FF = 512
EPOCHS = 5
BATCH = 32

def run_train(exp):
    out_dir = f"{OUT_DIR_BASE}/{exp['name']}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "python", TRAIN_PY,
        "--data_dir", DATA_DIR,
        "--vocab", str(VOCAB),
        "--d_model", str(D_MODEL),
        "--num_heads", str(NUM_HEADS),
        "--num_layers", str(NUM_LAYERS),
        "--d_ff", str(D_FF),
        "--batch", str(BATCH),
        "--epochs", str(EPOCHS),
        "--pos_encoding", exp["pos_encoding"],
        "--out_dir", out_dir,
        "--split"
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def read_csv_loss(path):
    epochs=[]; train=[]; val=[]
    with open(path,'r') as f:
        next(f)
        for ln in f:
            ep,tr,va = ln.strip().split(",")
            epochs.append(int(ep)); train.append(float(tr)); val.append(float(va))
    return epochs, train, val

def main():
    for exp in EXPERIMENTS:
        run_train(exp)

    plt.figure(figsize=(8,5))
    for exp in EXPERIMENTS:
        csvp = f"{OUT_DIR_BASE}/{exp['name']}/loss_{exp['pos_encoding']}.csv"
        if not os.path.exists(csvp): continue
        ep, tr, va = read_csv_loss(csvp)
        plt.plot(ep, tr, label=f"{exp['name']}_train")
        plt.plot(ep, va, label=f"{exp['name']}_val", linestyle='--')
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Train/Val loss comparison")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_compare.png")
    print("Saved plots/loss_compare.png")

if __name__ == "__main__":
    main()
