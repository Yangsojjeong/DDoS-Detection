import os, sys, argparse, math, time, json, re, hashlib
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import ipaddress

# ==== 하드코딩 설정 ==== #
BENIGN_CSV = r"C:\Users\zoezo\DDoS Detection\DataSet\dataset_normal_fixed_labeled.csv"
DDOS_CSV   = r"C:\Users\zoezo\DDoS Detection\DataSet\dataset_attack_fixed_labeled.csv"
OUTPUT_DIR = r"C:\Users\zoezo\DDoS Detection\preprocessing"
PACKETS_PER_FLOW = 10  # 패킷을 n개씩 묶어 하나의 샘플(플로우)로 사용
TIME_WINDOW = 32
DATASET_ID = "Last"
DEV_FRACTION = 0.8
DEV_SPLIT = (0.6, 0.2, 0.2)   # train, val, test

# ---------------- 유틸 ---------------- #
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명에서 BOM/앞뒤 공백 제거"""
    return df.rename(columns=lambda c: str(c).replace('\ufeff', '').strip())

def chunk_to_flows(mat, n_packets):
    """(N_packets, F) → (N_flows, n_packets, F). 마지막 조각은 0패딩."""
    flows = []
    for i in range(0, len(mat), n_packets):
        seg = mat[i:i+n_packets]
        if len(seg) == 0:
            break
        if len(seg) < n_packets:
            pad = np.zeros((n_packets - len(seg), mat.shape[1]), dtype=mat.dtype)
            seg = np.vstack([seg, pad])
        flows.append(seg)
    return np.array(flows)

def select_feature_columns(df, prefer_names=None):
    # 숫자형 컬럼만 사용(정답/라벨 후보 제외)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = set([c for c in df.columns if c.lower() in ['y','label','target','class']])
    cols = [c for c in num_cols if c not in skip]
    # prefer_names가 있으면 교집합 우선
    if prefer_names:
        pref = [c for c in prefer_names if c in cols]
        rest = [c for c in cols if c not in pref]
        return pref + rest
    return cols

def chunk_to_flows(mat, n_packets):
    flows = []
    for i in range(0, len(mat), n_packets):
        seg = mat[i:i+n_packets]
        if len(seg)==0: break
        if len(seg) < n_packets:
            pad = np.zeros((n_packets-len(seg), mat.shape[1]), dtype=mat.dtype)
            seg = np.vstack([seg, pad])
        flows.append(seg)
    return np.array(flows) # (num_flows, n_packets, F)

def normalize_0_1(arr3d, mins=None, maxs=None):
    # 열별 min–max 스케일링, train(dev)에서 구한 mins/maxs를 모든 split에 공통 적용.
    N,H,W = arr3d.shape
    flat = arr3d.reshape(-1, W)
    if mins is None or maxs is None:
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        # 상수열 보호
        maxs = np.where(maxs==mins, mins+1.0, maxs)
    norm = (flat - mins) / (maxs - mins)
    return norm.reshape(N,H,W), mins, maxs

def to_hdf5(X, y, out_path):
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('set_x', data=X)
        hf.create_dataset('set_y', data=y.astype(np.int8))

# ---------------- 메인 ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benign_csv',default=BENIGN_CSV)
    ap.add_argument('--ddos_csv', default=DDOS_CSV)
    ap.add_argument('--packets_per_flow', '-n', type=int, default=PACKETS_PER_FLOW, help='Use 1 for per-packet models')
    ap.add_argument('--time_window', '-t', type=int, default=TIME_WINDOW)
    ap.add_argument('--dataset_id', '-i', type=str, default=DATASET_ID)
    ap.add_argument('--output_folder', default=OUTPUT_DIR)
    # 기존: train_size=0.8, val_size=0.25 (→ 0.8×0.75=0.60 / 0.8×0.25=0.20 / 0.20)
    # 변경: dev_fraction + dev_split(train,val,test) + inference(1-dev_fraction)
    ap.add_argument('--dev_fraction', type=float, default=DEV_FRACTION, help='전체 중 학습용(dev) 비율 (나머지는 inference)')
    ap.add_argument('--dev_split', type=str, default='{}, {}, {}'.format(*DEV_SPLIT), help='dev 내부 train,val,test 비율 (합=1)')
    ap.add_argument('--feature_names', nargs='*', default=None)
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # 1) CSV 로드 + 컬럼 정리
    df_b = pd.read_csv(args.benign_csv, low_memory=False)
    df_d = pd.read_csv(args.ddos_csv,  low_memory=False)
    df_b = clean_columns(df_b)
    df_d = clean_columns(df_d)

    # 2) 피처 행렬 구성 (숫자형 + object 포함, Label 등 제외)
    feat_cols_b = select_feature_columns(df_b, args.feature_names)
    feat_cols_d = select_feature_columns(df_d, args.feature_names)
    feat_cols = [c for c in feat_cols_b if c in feat_cols_d]
    if len(feat_cols)==0:
        raise SystemExit("두 CSV 사이에 공통 숫자형 컬럼이 없습니다. feature_names 인자를 사용해 주세요.")
    Xb = df_b[feat_cols].to_numpy(dtype=np.float32)
    Xd = df_d[feat_cols].to_numpy(dtype=np.float32)

    # 3) 패킷 → 플로우 묶기
    n = int(args.packets_per_flow)
    Xb_flows = chunk_to_flows(Xb, n)
    Xd_flows = chunk_to_flows(Xd, n)

    # 파일 단위 라벨 (CSV의 Label 열은 피처에서 제외했지만, 이 코드는 사용하지 않고 파일 기준으로 y 생성)
    yb = np.zeros((Xb_flows.shape[0],), dtype=np.int8)  # benign=0
    yd = np.ones((Xd_flows.shape[0],), dtype=np.int8)   # attack=1


    # 4) 합치고 셔플 → train/val/test 분할
    X_all = np.vstack([Xb_flows, Xd_flows])   # (N, H, W)
    y_all = np.hstack([yb, yd])               # (N,)

    # -------- 새 분할 로직: dev vs inference 먼저, 그 다음 dev 내부에서 train/val/test --------
    dev_frac = float(args.dev_fraction)
    if not (0.0 < dev_frac < 1.0):
        raise SystemExit('--dev_fraction 은 (0,1) 사이 실수여야 합니다.')

    # dev_split 파싱
    try:
        tr_r, va_r, te_r = map(float, args.dev_split.split(','))
    except Exception:
        raise SystemExit('--dev_split 은 "0.6,0.2,0.2" 같은 형식이어야 합니다.')
    s = tr_r + va_r + te_r
    if not (abs(s - 1.0) < 1e-6):
        raise SystemExit('dev_split 의 합은 1이어야 합니다. 예: 0.6,0.2,0.2')

    # 먼저 dev / infer 분할 (stratify)
    X_dev, X_infer, y_dev, y_infer = train_test_split(
        X_all, y_all, train_size=dev_frac, stratify=y_all, shuffle=True, random_state=1
    )

    # dev 내부에서 train/val/test 분할
    # 1단계: dev → train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_dev, y_dev, train_size=tr_r, stratify=y_dev, shuffle=True, random_state=1
    )
    # 2단계: temp → val vs test (비율은 va_r : te_r 를 정규화)
    if (va_r + te_r) <= 0:
        raise SystemExit('dev_split 의 val,test 비율이 0이면 안 됩니다.')
    val_rel = va_r / (va_r + te_r)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_rel, stratify=y_temp, shuffle=True, random_state=1
    )

    # 5) 정규화: dev(train+val+test)에 대해서만 min/max 산출 → 모든 split에 동일 적용
    X_dev_full = np.vstack([X_train, X_val, X_test])  # (N_dev, H, W)
    X_dev_norm, mins, maxs = normalize_0_1(X_dev_full)  # dev 기준으로 min/max 계산
    a, b = X_train.shape[0], X_val.shape[0]
    X_train_n = X_dev_norm[:a]
    X_val_n   = X_dev_norm[a:a+b]
    X_test_n  = X_dev_norm[a+b:]

    # inference 세트에도 dev에서 얻은 mins/maxs 적용
    X_infer_n, _, _ = normalize_0_1(X_infer, mins=mins, maxs=maxs)

    # 6) 저장 파일명
    prefix = f"{args.time_window}t-{args.packets_per_flow}n-{args.dataset_id}-dataset"
    out_train = os.path.join(args.output_folder, prefix + '-train.hdf5')
    out_val   = os.path.join(args.output_folder, prefix + '-val.hdf5')
    out_test  = os.path.join(args.output_folder, prefix + '-test.hdf5')
    out_infer = os.path.join(args.output_folder, prefix + '-infer.hdf5')  # 새로 추가

    to_hdf5(X_train_n, y_train, out_train)
    to_hdf5(X_val_n,   y_val,   out_val)
    to_hdf5(X_test_n,  y_test,  out_test)
    to_hdf5(X_infer_n, y_infer, out_infer)  # 새로 추가

    # 스케일러(JSON): dev 기준 mins/maxs 와 feature_cols 저장
    scaler_path = os.path.join(args.output_folder, prefix + '-scaler.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump({
            'feature_cols': feat_cols,
            'mins': mins.tolist(),
            'maxs': maxs.tolist(),
            'note': 'Use with realtime_infer_csv.py; feature names must match.'
        }, f, ensure_ascii=False, indent=2)

    # 7) 로그(summary) - inference 포함
    summary = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'examples': {'tot': int(y_all.shape[0]),
                     'ben': int((y_all==0).sum()),
                     'ddos': int((y_all==1).sum())},
        'fractions': {
            'dev_fraction': dev_frac,
            'dev_split': {'train': tr_r, 'val': va_r, 'test': te_r},
            'inference_fraction': round(1.0 - dev_frac, 6)
        },
        'sizes': {
            'train': int(y_train.shape[0]),
            'val':   int(y_val.shape[0]),
            'test':  int(y_test.shape[0]),
            'infer': int(y_infer.shape[0])
        },
        'shape': {'H': int(n), 'W': int(X_all.shape[2])},
        'feature_cols': feat_cols,
        'files': {
            'train': out_train, 'val': out_val, 'test': out_test, 'infer': out_infer,
            'scaler': scaler_path
        }
    }
    with open(os.path.join(args.output_folder, prefix + '-summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Wrote:', out_train)
    print('Wrote:', out_val)
    print('Wrote:', out_test)
    print('Wrote:', out_infer)
    print('Wrote:', scaler_path)

if __name__ == '__main__':
    main()
