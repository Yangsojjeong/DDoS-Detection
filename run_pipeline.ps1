Param(
  [string]$DatasetName = "ANOTHER",
  [string]$BenignCsv   = ".\another-dataset\dataset_normal.csv",
  [string]$DdosCsv     = ".\another-dataset\dataset_ddos.csv",
  [int]$PacketsPerFlow = 10,
  [int]$TimeWindow     = 10,
  [float]$TrainSize    = 0.7,
  [float]$ValSize      = 0.15,
  [int]$Epochs         = 200
)

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zoezo\DDoS Detection"

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$RUN = ".\runs\$ts"
$PRE = "$RUN\work\pre"
$OUT = "$RUN\output"

New-Item -ItemType Directory -Force $RUN,$PRE,$OUT | Out-Null

# 1) CSV -> HDF5
.\.venv\Scripts\python.exe .\csv_to_hdf5.py `
  --benign_csv "$BenignCsv" `
  --ddos_csv "$DdosCsv" `
  --output_folder "$PRE" `
  --packets_per_flow $PacketsPerFlow `
  --time_window $TimeWindow `
  --dataset_id $DatasetName `
  --train_size $TrainSize `
  --val_size $ValSize 2>&1 | Tee-Object -FilePath "$OUT\preprocess-console.log"

$PREFIX = "${TimeWindow}t-${PacketsPerFlow}n-$DatasetName"
$TRAIN_H5 = Join-Path $PRE "$PREFIX-dataset-train.hdf5"
$VAL_H5   = Join-Path $PRE "$PREFIX-dataset-val.hdf5"
$TEST_H5  = Join-Path $PRE "$PREFIX-dataset-test.hdf5"

# 2) Train
.\.venv\Scripts\python.exe .\lucid_cnn.py `
  --train "$PRE" `
  --epochs $Epochs `
  --output_dir "$OUT" `
  --prefix "$PREFIX" 2>&1 | Tee-Object -FilePath "$OUT\train-console.log"

$BEST = Get-Content "$OUT\best_model_path.txt"

# 3) Predict
.\.venv\Scripts\python.exe .\lucid_cnn.py `
  --predict "$PRE" `
  --model "$BEST" `
  --output_dir "$OUT" `
  --prefix "$PREFIX" 2>&1 | Tee-Object -FilePath "$OUT\predict-console.log"

# 4) Visualize
.\.venv\Scripts\python.exe .\visualize_train.py `
  --trainlog "$OUT\$PREFIX-LUCID-trainlog.csv" `
  --output_dir "$OUT"

.\.venv\Scripts\python.exe .\visualize_eval_summary.py `
  --pred_csv_auto "$OUT" `
  --output_dir "$OUT"

Write-Host "RUN folder: $RUN"