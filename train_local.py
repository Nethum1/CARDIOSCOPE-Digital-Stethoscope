# ================================================================
# train_local.py — CardioScope Model Training
#
# Matches ESP32 firmware:
#   Sample rate : 800 SPS
#   Duration    : 5 seconds per clip
#   n_fft       : 512  (safe for short/low-SR audio)
#   hop_length  : 128
#
# Dataset folder structure:
#   dataset/
#   ├── normal/         ← .wav files of normal heart sounds
#   ├── murmur/         ← .wav files with murmur
#   
#        
#
# Recommended dataset: PhysioNet 2016
#   https://physionet.org/content/challenge-2016/
#
# Run: python train_local.py
# Output: models/stethoscope_model.pkl
#         models/scaler.pkl
#         models/label_encoder.pkl
# ================================================================

import numpy as np
import os
import joblib
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (classification_report,
                                      accuracy_score,
                                      confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# MUST MATCH ESP32 FIRMWARE AND SERVER
# ================================================================
SAMPLE_RATE    = 800    # Hz  — matches ESP32 ADS1115 rate
DURATION       = 5.0    # seconds — matches RECORD_SECONDS in firmware
N_MFCC         = 40     # MFCC coefficients
N_FFT          = 512    # safe for 800 SPS × 5s = 4000 samples
HOP_LENGTH     = 128    # n_fft / 4

DATASET_DIR = 'dataset'
MODEL_DIR   = 'models'
CLASSES     = ['normal', 'murmur']

# ================================================================
# FEATURE EXTRACTION
# IDENTICAL to local_prediction_server.py extract_all()
# ================================================================
def extract_features(audio_path):
    """
    Load a WAV file and extract the ~95-value feature vector.
    Resamples to SAMPLE_RATE and pads/trims to DURATION.
    """
    # Load and resample to our target rate
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE,
                          duration=DURATION)

    # Pad if shorter than DURATION
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    # Trim to exact length
    y = y[:target_len]

    # ── MFCC (main feature) ──────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )  # (40, T)

    # ── Supporting features ──────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec_c = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr    = librosa.feature.zero_crossing_rate(
        y, hop_length=HOP_LENGTH)
    rms    = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    # ── Feature vector ───────────────────────────────────────
    features = np.concatenate([
        np.mean(mfcc,   axis=1),   # 40
        np.std (mfcc,   axis=1),   # 40
        np.mean(chroma, axis=1),   # 12
        [np.mean(spec_c)],          #  1
        [np.mean(zcr)],             #  1
        [np.mean(rms)],             #  1
    ])  # 95 total

    return features

# ================================================================
# LOAD DATASET
# ================================================================
def load_dataset():
    X, y, files_ok, files_err = [], [], 0, 0
    print(f"\nLoading dataset from '{DATASET_DIR}/'")
    print(f"Settings: {SAMPLE_RATE} SPS, {DURATION}s, n_fft={N_FFT}")
    print("-" * 50)

    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.exists(folder):
            print(f"  [SKIP] {label}/ not found")
            continue

        wav_files = [f for f in os.listdir(folder)
                     if f.lower().endswith('.wav')]
        print(f"  [{label}] {len(wav_files)} files")

        for fname in wav_files:
            path = os.path.join(folder, fname)
            try:
                feat = extract_features(path)
                X.append(feat)
                y.append(label)
                files_ok += 1
            except Exception as e:
                files_err += 1
                print(f"    [ERR] {fname}: {e}")

    print("-" * 50)
    print(f"Loaded  : {files_ok} files OK  |  {files_err} errors")
    return np.array(X), np.array(y)

# ================================================================
# TRAIN
# ================================================================
def train():
    X, y = load_dataset()

    if len(X) == 0:
        print("\n[ERROR] No data found!")
        print("Place .wav files in:")
        for c in CLASSES:
            print(f"  {DATASET_DIR}/{c}/")
        return

    print(f"\nTotal samples: {len(X)}")
    print(f"Classes found: {sorted(set(y))}")
    print(f"Feature vector size: {X.shape[1]}")

    # Check class balance
    print("\nClass distribution:")
    for c in sorted(set(y)):
        n = np.sum(y == c)
        print(f"  {c:<15} {n:>4} samples")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\nEncoded labels: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42
    )
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test : {len(X_test)} samples")

    # ── Train Random Forest ──────────────────────────────────
    print("\nTraining Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',  # handles uneven class counts
        random_state=42,
        n_jobs=-1                 # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("Training complete!")

    # ── Evaluate ─────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"  TEST ACCURACY: {acc:.2%}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("            " + "  ".join(f"{c[:6]:>6}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<12}" +
              "  ".join(f"{v:>6}" for v in row))

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  f'{MODEL_DIR}/stethoscope_model.pkl')
    joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
    joblib.dump(le,     f'{MODEL_DIR}/label_encoder.pkl')

    print(f"\n[SAVED] {MODEL_DIR}/stethoscope_model.pkl")
    print(f"[SAVED] {MODEL_DIR}/scaler.pkl")
    print(f"[SAVED] {MODEL_DIR}/label_encoder.pkl")
    print("\nModel is ready. Start local_prediction_server.py")

if __name__ == '__main__':
    train()
