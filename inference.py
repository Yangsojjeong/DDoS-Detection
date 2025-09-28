import os
import sys
import time
import json
import argparse
import csv
import numpy as np
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #tensorflow log level 설정 내용

def load_scaler_json(path): #scaler_json 파일을 불러와야함
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    mins = np.array(obj["mins"], dtype=np.float32)
    maxs = np.array(obj["maxs"], dtype=np.float32)
    maxs = np.where(maxs == mins, mins + 1.0, maxs)
    return mins, maxs

def normalize_array(arr, mins, maxs): #정규화
    x = (arr - mins) / (maxs - mins)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r"C:\Users\zoezo\DDoS Detection\model&result\32t-10n-Last-Last-best.h5") #모델
    parser.add_argument("--hdf5", type=str, default=r"C:\Users\zoezo\DDoS Detection\model&result\32t-10n-Last-dataset-infer.hdf5")
    parser.add_argument("--x_key", type=str, default="set_x")
    parser.add_argument("--y_key", type=str, default="set_y")
    parser.add_argument("--scaler_json", type=str, default=r"C:\Users\zoezo\DDoS Detection\model&result\32t-10n-Last-dataset-train_norm.hdf5")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default=r"C:\Users\zoezo\DDoS Detection\model&result\Last_hdf5_predictions.csv")
    args = parser.parse_args()

    mins, maxs = load_scaler_json(args.scaler_json)
    model = load_model(args.model, compile=False) #추론만 하므로 compile 불필요

    with h5py.File(args.hdf5, "r") as f:
        X = f[args.x_key] 
        Y = f[args.y_key] if args.y_key in f else None

        N = X.shape[0]
        x_ndim = X.ndim #입력 데이터 차원(2,3,4 중 하나만 지원)

        if x_ndim not in [2, 3, 4]:
            raise ValueError(f"Unsupported input shape: {X.shape}")

        preds, probs, labels = [], [], [] #예측 레이블, 확률, 정답 label
    
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["index", "pred", "prob_attack", "y_true", "frame.len", "ip.proto"])

            for i in range(N):
                x = X[i]  # i번째 샘플 읽기

                if x_ndim == 4:
                    x = np.squeeze(x, axis=-1)
                if x_ndim in [2, 3]:
                    H, W = x.shape[-2], x.shape[-1]
                    if mins.shape[0] != W:
                        raise ValueError(f"Scaler mismatch: W={W} vs mins={mins.shape[0]}")
                    x_norm = normalize_array(x, mins, maxs)[np.newaxis, ...]
                else:
                    W = x.shape[0]
                    x_norm = normalize_array(x, mins, maxs)[np.newaxis, ...]

                prob = float(model.predict(x_norm, verbose=0).ravel()[0])
                pred = int(prob >= args.score_threshold)
                y_true = int(np.ravel(Y[i])[0]) if Y is not None else None

                # ---- 여기서 feature 값 추출 ----
                frame_len_val = float(np.ravel(x[..., 1])[0])   # "frame.len"
                ip_proto_val  = float(np.ravel(x[..., 9])[0])   # "ip.proto"
                # --------------------------------

                preds.append(pred)
                probs.append(prob)
                labels.append(y_true)

                print(f"[Sample {i}] prob_attack={prob:.4f} | pred={pred} | y_true={y_true} "
                      f"| frame.len={frame_len_val} | ip.proto={ip_proto_val}", flush=True)

                writer.writerow([i, pred, prob, y_true, frame_len_val, ip_proto_val])
                fout.flush()

    # ----- F1 Score 출력 -----
    if all(l is not None for l in labels):
        f1 = f1_score(labels, preds)
        print(f"[INFO] Final F1 Score: {f1:.4f}", flush=True)
    else:
        print("[INFO] Ground truth labels(Y) 없음 → F1 Score 계산 불가", flush=True)

    print(f"[INFO] Saved {len(preds)} rows to {args.out_csv}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
