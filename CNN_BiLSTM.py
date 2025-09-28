import os, sys, glob, time, re, csv, pprint
import numpy as np
import random as rn
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Activation, SpatialDropout1D, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import shuffle
from tensorflow.keras.metrics import AUC
from util_functions import SEED, load_dataset, count_packets_in_dataset
from tensorflow.keras.layers import GlobalAveragePooling1D 


# reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

K.set_image_data_format('channels_last')
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 출력 헤더
VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score',
                  'TPR', 'FPR','TNR', 'FNR', 'Source']

# 기본 세팅
PATIENCE = 100
DEFAULT_EPOCHS = 100

# confusion matrix & summary 저장
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_confusion_and_summary_min(out_dir, y_true, y_pred, make_csv=True):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # 1) confusion matrix 이미지
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix'); plt.colorbar()
    plt.xticks([0,1], ['Benign','DDoS']); plt.yticks([0,1], ['Benign','DDoS'])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred-confusion_matrix.png"), dpi=150)
    plt.close()

    # 2) 텍스트 요약
    with open(os.path.join(out_dir, "eval-summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")
        f.write(f"Accuracy={acc:.4f}\nPrecision={prec:.4f}\nRecall={rec:.4f}\nF1={f1:.4f}\n")

    # 3) 막대그래프 요약
    plt.figure()
    metrics = ['Accuracy','Precision','Recall','F1']
    vals = [acc,prec,rec,f1]
    plt.bar(metrics, vals)
    plt.ylim(0,1)
    plt.title('Evaluation Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval-summary.png"), dpi=150)
    plt.close()

    # 4) CSV 요약
    if make_csv:
        with open(os.path.join(out_dir, "pred-summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(['TP','TN','FP','FN','Accuracy','Precision','Recall','F1'])
            w.writerow([tp,tn,fp,fn,acc,prec,rec,f1])


# ===  CNN+BiLSTM 모델 === #
def CNN_BiLSTM_v2(input_shape):
    model = Sequential(name="CNN_BiLSTM_v2")

    # ---------- CNN Blocks (no aggressive downsampling) ----------
    # Block 1: 기본 특징 추출 + 1회 다운샘플(길이 10 → 5)
    model.add(Conv1D(64, kernel_size=8, padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))   # 시퀀스 길이: 10 → 5

    # Block 2: 확장 컨볼루션(dilated)로 더 넓은 문맥
    model.add(Conv1D(64, kernel_size=5, dilation_rate=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Block 3: 얕은 커널로 미세 패턴
    model.add(Conv1D(32, kernel_size=3, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 과적합/특정 채널 의존 완화
    model.add(SpatialDropout1D(0.2))

    # ---------- Recurrent (BiLSTM) ----------
    # 길이 5 시퀀스를 양방향으로 처리
    model.add(Bidirectional(LSTM(64)))   # 출력: (None, 128)
    model.add(Dropout(0.5))

    # ---------- Dense Head ----------
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # ---------- Compile ----------
    opt = Adam(learning_rate=3e-4, clipnorm=1.0)  # 폭주 방지
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return model


def main(argv):
    parser = argparse.ArgumentParser(
        description='CNN-based DDoS detection (IoT Defence CNN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')
    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')
    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output dir')
    parser.add_argument('--minimal_output', action='store_true',
                        help='Save only essential artifacts')
    args = parser.parse_args()

    RUN_DIR = os.environ.get("RUN", ".")
    OUTPUT_FOLDER = args.output_dir or os.path.join(RUN_DIR, "output")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # === TRAIN ===
    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0:
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)

        for dataset_folder in subfolders:
            X_train, Y_train = load_dataset(dataset_folder + "/*-train.hdf5")
            X_val,   Y_val   = load_dataset(dataset_folder + "/*-val.hdf5")

            # Conv1D 입력으로 맞춤
            X_train = X_train.squeeze(-1)
            X_val   = X_val.squeeze(-1)

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val,   Y_val   = shuffle(X_val, Y_val, random_state=SEED)

            # 파일명 파싱
            train_file = glob.glob(os.path.join(dataset_folder, "*-train.hdf5"))[0]
            filename = os.path.basename(train_file).strip()
            parts = filename.split('-')
            time_window = int(parts[0].replace('t',''))
            max_flow_len = int(parts[1].replace('n',''))
            dataset_name = parts[2].strip()

            model_name = dataset_name + "-Last"
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = CNN_BiLSTM_v2(input_shape)

            best_model_filename = os.path.join(
                OUTPUT_FOLDER, f"{time_window}t-{max_flow_len}n-{model_name}"
            )

            es = EarlyStopping(monitor='val_loss', mode='min',
                               verbose=1, patience=PATIENCE)
            csvlog = CSVLogger(best_model_filename + '-trainlog.csv')
            mc = ModelCheckpoint(best_model_filename + '-best.h5',
                                 monitor='val_accuracy',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)

            model.fit(X_train, Y_train,
                      epochs=args.epochs,
                      validation_data=(X_val, Y_val),
                      callbacks=[es, mc, csvlog],
                      batch_size=32)

            print("Best model path:", best_model_filename + "-best.h5")

    # === PREDICT ===
    if args.predict is not None:
        predict_file = open(os.path.join(OUTPUT_FOLDER, 'predictions-' +
                                         time.strftime("%Y%m%d-%H%M%S") + '.csv'),
                            'w', newline='')
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader(); predict_file.flush()

        dataset_filelist = glob.glob(os.path.join(args.predict, "*-test.hdf5"))
        if not dataset_filelist:
            raise FileNotFoundError(f"No *-test.hdf5 found in {args.predict}")

        model_list = [args.model] if args.model else glob.glob(os.path.join(args.predict, "*-best.h5"))
        if not model_list:
            raise FileNotFoundError("No model found. Use --model to specify one.")

        for model_path in model_list:
            model = load_model(model_path)

            for ds_path in dataset_filelist:
                X, Y = load_dataset(ds_path)
                X = X.squeeze(-1)

                Y_pred = (model.predict(X, batch_size=2048) > 0.5).astype(int).ravel()
                [packets] = count_packets_in_dataset([X])

                if Y is not None:
                    _save_confusion_and_summary_min(OUTPUT_FOLDER, Y, Y_pred)

                tn, fp, fn, tp = confusion_matrix(Y, Y_pred, labels=[0,1]).ravel()
                tpr = tp / (tp + fn) if (tp+fn)>0 else 0
                fpr = fp / (fp + tn) if (fp+tn)>0 else 0
                tnr = tn / (tn + fp) if (tn+fp)>0 else 0
                fnr = fn / (fn + tp) if (fn+tp)>0 else 0
                acc = accuracy_score(Y, Y_pred)
                f1  = f1_score(Y, Y_pred)

                row = {'Model': os.path.basename(model_path),
                       'Time': '{:04.3f}'.format(0.0),
                       'Packets': packets,
                       'Samples': Y_pred.shape[0],
                       'DDOS%': '{:04.3f}'.format(sum(Y_pred)/len(Y_pred)),
                       'Accuracy': '{:05.4f}'.format(acc),
                       'F1Score': '{:05.4f}'.format(f1),
                       'TPR': '{:05.4f}'.format(tpr),
                       'FPR': '{:05.4f}'.format(fpr),
                       'TNR': '{:05.4f}'.format(tnr),
                       'FNR': '{:05.4f}'.format(fnr),
                       'Source': os.path.basename(ds_path)}
                pprint.pprint(row, sort_dicts=False)
                predict_writer.writerow(row)
                predict_file.flush()

        predict_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
