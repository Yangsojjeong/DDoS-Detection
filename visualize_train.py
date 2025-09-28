# visualize_train.py
import os, argparse, pandas as pd, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--trainlog", type=str, required=True, help="Path to *-trainlog.csv")
parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save plots")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv(args.trainlog)
epochs = df['epoch'] if 'epoch' in df else range(len(df))

plt.figure()
if 'loss' in df.columns: plt.plot(epochs, df['loss'], label='loss')
if 'val_loss' in df.columns: plt.plot(epochs, df['val_loss'], label='val_loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss Curves')
plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'train_loss.png'), dpi=160)

if 'accuracy' in df.columns or 'val_accuracy' in df.columns:
    plt.figure()
    if 'accuracy' in df.columns: plt.plot(epochs, df['accuracy'], label='accuracy')
    if 'val_accuracy' in df.columns: plt.plot(epochs, df['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy Curves')
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, 'train_accuracy.png'), dpi=160)

print(f"Saved: {os.path.join(args.output_dir, 'train_loss.png')}")
print(f"Saved: {os.path.join(args.output_dir, 'train_accuracy.png')}")
