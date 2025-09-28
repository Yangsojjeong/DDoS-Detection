import h5py
import numpy as np
import json

# 입력 HDF5 파일 경로
hdf5_path = r"C:/Users/zoezo/DDoS Detection/model&result/32t-10n-Last-dataset-train.hdf5"

# 출력 JSON 파일 경로
json_path = r"C:/Users/zoezo\DDoS Detection/model&result/32t-10n-Last-dataset-train_norm.hdf5"

# 실제 feature 이름 
feature_cols = [
     "frame.encap_type",
    "frame.len",
    "frame.protocols",
    "ip.dst",
    "ip.flags.df",
    "ip.flags.rb",
    "ip.frag_offset",
    "ip.hdr_len",
    "ip.len",
    "ip.proto",
    "ip.src",
    "ip.ttl",
    "p.flags.mf",
    "tcp.ack",
    "tcp.dstport",
    "tcp.flags.ack",
    "tcp.flags.cwr",
    "tcp.flags.ecn",
    "tcp.flags.fin",
    "tcp.flags.ns",
    "tcp.flags.push",
    "tcp.flags.res",
    "tcp.flags.reset",
    "tcp.flags.syn",
    "tcp.flags.urg",
    "tcp.len",
    "tcp.srcport",
    "tcp.time_delta",
    "tcp.window_size"
]
# HDF5 파일 열기
with h5py.File(hdf5_path, "r") as f:
    X = f["set_x"][:]  # 입력 데이터 전체 불러오기
    
    # 마지막 축이 feature dimension
    n_features = X.shape[-1]
    if n_features != len(feature_cols):
        raise ValueError(f"Feature 개수 불일치: HDF5={n_features}, feature_cols={len(feature_cols)}")
    
    # feature-wise min/max 계산
    mins = np.min(X.reshape(-1, n_features), axis=0).tolist()
    maxs = np.max(X.reshape(-1, n_features), axis=0).tolist()

# JSON 객체 구성
scaler_obj = {
    "feature_cols": feature_cols,
    "mins": mins,
    "maxs": maxs,
    "note": "Use with realtime_infer_csv.py; feature names must match."
}

# JSON 저장
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(scaler_obj, f, indent=2, ensure_ascii=False)

print(f"[INFO] JSON 파일 생성 완료: {json_path}")
