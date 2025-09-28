# visualize_eval_summary.py
import os, argparse, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--pred_csv", type=str, help="Path to predictions-*.csv")
parser.add_argument("--pred_csv_auto", type=str, help="Directory to search predictions-*.csv (use latest)")
parser.add_argument("--output_dir", type=str, default="./output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pred_path = args.pred_csv
if pred_path is None and args.pred_csv_auto:
    candidates = sorted(glob.glob(os.path.join(args.pred_csv_auto, 'predictions-*.csv')))
    if not candidates:
        raise FileNotFoundError("No predictions-*.csv found in " + args.pred_csv_auto)
    pred_path = candidates[-1]

if pred_path is None:
    raise SystemExit("Provide --pred_csv or --pred_csv_auto")

df = pd.read_csv(pred_path)
row = df.iloc[0].to_dict()

samples = int(float(row.get('Samples', 0) or 0))
ddos_ratio = float(row.get('DDOS%', 0) or 0)
acc = float(row.get('Accuracy', 0) or 0)
f1 = float(row.get('F1Score', 0) or 0)
tpr = float(row.get('TPR', 0) or 0)
fpr = float(row.get('FPR', 0) or 0)
tnr = float(row.get('TNR', 0) or 0)
fnr = float(row.get('FNR', 0) or 0)

# 1) Summary bar
plt.figure()
metrics = ['Accuracy','F1','TPR','TNR','FPR','FNR']
values = [acc, f1, tpr, tnr, fpr, fnr]
plt.bar(metrics, values)
plt.ylim(0, 1.0)
plt.title('Evaluation Summary')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'eval_summary.png'), dpi=160)

# 2) Approx confusion matrix
pos = int(round(samples * ddos_ratio))
neg = max(0, samples - pos)
tp = int(round(pos * tpr))
fn = max(0, pos - tp)
tn = int(round(neg * tnr))
fp = max(0, neg - tn)

fig = plt.figure()
mat = np.array([[tn, fp],[fn, tp]])
plt.imshow(mat)
plt.xticks([0,1], ['Pred 0','Pred 1'])
plt.yticks([0,1], ['True 0','True 1'])
for (i,j), v in np.ndenumerate(mat):
    plt.text(j, i, str(v), ha='center', va='center')
plt.title('Confusion Matrix (approx.)')
plt.colorbar()
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'confusion_matrix_approx.png'), dpi=160)

# Also export a small CSV summary
out_csv = os.path.join(args.output_dir, 'eval-summary.csv')
pd.DataFrame([{
    'Samples': samples, 'DDOS%': ddos_ratio,
    'Accuracy': acc, 'F1Score': f1, 'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'FNR': fnr
}]).to_csv(out_csv, index=False)

print('Saved:', os.path.join(args.output_dir, 'eval_summary.png'))
print('Saved:', os.path.join(args.output_dir, 'confusion_matrix_approx.png'))
print('Saved:', out_csv)
