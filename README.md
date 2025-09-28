# 🛡️ DDoS Detection with CNN+BiLSTM

This project implements a deep learning-based model for **DDoS attack detection** using network traffic data.  
Raw packet/flow data is preprocessed into HDF5 format, and a hybrid **CNN + BiLSTM** model is trained, evaluated, and used for inference.

---

## 📂 Project Structure
```
.
├── preprocessing.py              # Preprocess raw CSV into HDF5/JSON
├── make_hdf5.py                   # Create HDF5 datasets + extract normalization parameters
├── dataset_parser.py              # Parse HDF5, feature scaling utilities
├── util_functions.py              # Common utilities (seeding, normalization, etc.)
├── CNN_BiLSTM.py                  # CNN+BiLSTM model definition & training
├── inference.py                   # Run inference on HDF5 datasets with trained model
├── visualize_train.py             # Visualize training logs (*.csv)
├── visualize_eval_summary.py      # Summarize and plot inference results
├── model&result/                  # Saved models (.h5), logs, predictions, plots
└── preprocessing data/            # Preprocessed datasets (.hdf5, scaler.json, summary.json)
```

---

## ⚙️ Setup
```bash
# Create and activate virtual environment (Windows PowerShell example)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Data Preprocessing
1. **Convert CSV → HDF5**
   ```bash
   python preprocessing.py --input dataset_attack.csv --output "preprocessing data/"
   python preprocessing.py --input dataset_normal.csv --output "preprocessing data/"
   ```

2. **Extract normalization parameters (min/max)**
   ```bash
   python make_hdf5.py
   ```
   → Generates `*-scaler.json` (used during inference)

---

## 🧠 Model Training
Train the CNN+BiLSTM model:

```bash
python CNN_BiLSTM.py --train "preprocessing data" --epochs 100 --output_dir model&result
```

- Logs: `*-trainlog.csv`  
- Best model: `*-best.h5` (via ModelCheckpoint)  
- Early stopping enabled  

---

## 🔍 Inference
Run inference with a trained model:

```bash
python inference.py   --model model&result/32t-10n-Last-Last-best.h5   --hdf5 "preprocessing data/32t-10n-Last-dataset-infer.hdf5"   --scaler_json "preprocessing data/32t-10n-Last-dataset-scaler.json"   --out_csv model&result/Last_hdf5_predictions.csv
```

**Output:**
- Console log: predictions (`pred`, `prob_attack`)  
- CSV file: `index, pred, prob_attack, y_true, frame.len, ip.proto`  

---

## 📊 Visualization
1. **Training Process**
   ```bash
   python visualize_train.py      --trainlog model&result/32t-10n-Last-Last-trainlog.csv      --output_dir model&result
   ```
   → `train_loss.png`, `train_accuracy.png`

2. **Evaluation Summary**
   ```bash
   python visualize_eval_summary.py      --pred_csv model&result/predictions-*.csv      --output_dir model&result
   ```
   → `eval_summary.png`, `confusion_matrix_approx.png`, `eval-summary.csv`

---

## 📌 Key Features
- **CNN blocks** for feature extraction with dilation convolution  
- **BiLSTM** for sequential flow dependencies  
- **Normalization JSON** for real-time inference consistency  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC  

---

## 🚀 Future Work
- Real-time streaming packet inference  
- Integration with reinforcement learning for **network slicing & resource allocation**  
- Benchmarking on multiple DDoS datasets  
