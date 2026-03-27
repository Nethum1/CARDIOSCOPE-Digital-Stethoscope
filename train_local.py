# ================================================================
# train_local.py  — CardioScope Model Trainer  v2.0
#
# ROOT CAUSE FIX:
#   Downloaded datasets are 44100 Hz (or 22050 Hz).
#   ESP32 records at 800 SPS.
#   Old model trained on 44100 Hz features → always predicts NORMAL
#   because it never saw 800 Hz features during training.
#
#   THIS script resamples EVERY audio file to 800 Hz before
#   extracting features — so training exactly matches inference.
#
# FOLDER STRUCTURE REQUIRED:
#   data/
#     normal/   ← put normal heart sound .wav files here
#     murmur/   ← put murmur .wav files here
#   models/     ← created automatically, model saved here
#
# RECOMMENDED DATASETS (free):
#   PhysioNet 2016 Challenge  — https://physionet.org/content/challenge-2016/
#   Pascal Heart Sound        — search "pascal heart sound dataset"
#   Each has normal + abnormal (murmur) .wav files
#
# Run: python train_local.py
# ================================================================

import os
import sys
import numpy as np
import librosa
import joblib
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import classification_report, confusion_matrix
from sklearn.pipeline          import Pipeline
from collections               import Counter
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# MUST MATCH ESP32 FIRMWARE AND local_prediction_server.py
# ================================================================
TARGET_SR       = 800    # resample everything to this — matches ESP32
RECORD_SECONDS  = 5      # crop / pad to this length
TARGET_SAMPLES  = TARGET_SR * RECORD_SECONDS  # 4000 samples

DATA_DIR    = 'dataset'
MODELS_DIR  = 'models'
CLASSES     = ['Normal', 'Murmur']   # must match your folder names exactly

# ================================================================
# FEATURE EXTRACTION  — identical to local_prediction_server.py
# CRITICAL: any change here must be mirrored in the server too
# ================================================================
def extract_features(audio_float, sample_rate=TARGET_SR):
    y  = audio_float
    sr = sample_rate

    n_fft      = min(512, len(y))
    hop_length = n_fft // 4

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                   n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=n_fft, hop_length=hop_length)
    spec_c = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                n_fft=n_fft, hop_length=hop_length)
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms    = librosa.feature.rms(y=y, hop_length=hop_length)

    features = np.concatenate([
        np.mean(mfcc,   axis=1),   # 40
        np.std (mfcc,   axis=1),   # 40
        np.mean(chroma, axis=1),   # 12
        [np.mean(spec_c)],          #  1
        [np.mean(zcr)],             #  1
        [np.mean(rms)],             #  1
    ])                              # = 95 total

    return features

# ================================================================
# LOAD + RESAMPLE ONE FILE  ← THE KEY FIX
# ================================================================
def load_and_resample(filepath):
    """
    Load any audio file (any sample rate) and:
      1. Resample to TARGET_SR (800 Hz) — matches ESP32
      2. Convert to mono
      3. Crop or pad to exactly TARGET_SAMPLES (4000 samples = 5s)
    Returns float32 array of shape (4000,)
    """
    try:
        # sr=None preserves original rate; mono=True converts stereo
        y, orig_sr = librosa.load(filepath, sr=None, mono=True)

        # ── RESAMPLE to 800 Hz ──────────────────────────────────
        # This is the critical step that was missing before.
        # Training at 44100 Hz → model never saw 800 Hz features.
        if orig_sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=TARGET_SR)

        # ── Normalise amplitude ─────────────────────────────────
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.9

        # ── Crop or pad to exactly TARGET_SAMPLES ───────────────
        if len(y) >= TARGET_SAMPLES:
            # Take the middle section — avoids silent start/end
            start = (len(y) - TARGET_SAMPLES) // 2
            y = y[start : start + TARGET_SAMPLES]
        else:
            # Tile the signal to fill 5 seconds (loop short clips)
            repeats = (TARGET_SAMPLES // len(y)) + 1
            y = np.tile(y, repeats)[:TARGET_SAMPLES]

        return y.astype(np.float32)

    except Exception as e:
        print(f"    [SKIP] {os.path.basename(filepath)}: {e}")
        return None

# ================================================================
# DATA AUGMENTATION — triples the dataset size
# Helps when you have few recordings per class
# All augmentations stay at 800 Hz
# ================================================================
def augment(y, sr=TARGET_SR):
    augmented = []

    # 1. Add gentle background noise
    noise = np.random.randn(len(y)).astype(np.float32) * 0.005
    augmented.append(y + noise)

    # 2. Pitch shift ±1 semitone (simulates different chest sizes)
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))
    except Exception:
        pass

    # 3. Small time stretch (±10%)
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=1.1)[:len(y)])
        augmented.append(librosa.effects.time_stretch(y, rate=0.9)[:len(y)])
    except Exception:
        pass

    # 4. Volume variation (±20%)
    augmented.append(y * 1.2)
    augmented.append(y * 0.8)

    return augmented

# ================================================================
# LOAD DATASET
# ================================================================
def load_dataset(use_augmentation=True):
    X, y_labels = [], []
    class_counts = {}

    print(f"\n{'═'*52}")
    print(f"  Loading audio → resampling to {TARGET_SR} Hz")
    print(f"{'═'*52}")

    for label in CLASSES:
        class_dir = os.path.join(DATA_DIR, label)

        if not os.path.isdir(class_dir):
            print(f"\n[ERROR] Missing folder: {class_dir}")
            print(f"        Create it and add .wav files for class '{label}'")
            sys.exit(1)

        # Support wav, mp3, flac, ogg, aiff
        audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif')
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(audio_exts)]

        if not files:
            print(f"\n[ERROR] No audio files in {class_dir}")
            print(f"        Add .wav files and re-run.")
            sys.exit(1)

        print(f"\n  [{label.upper()}] {len(files)} files found")

        ok_count = 0
        for fname in sorted(files):
            fpath = os.path.join(class_dir, fname)
            audio = load_and_resample(fpath)

            if audio is None:
                continue

            # Extract features from original
            feat = extract_features(audio)
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                print(f"    [SKIP] {fname}: NaN/Inf in features")
                continue

            X.append(feat)
            y_labels.append(label)
            ok_count += 1
            print(f"    ✓ {fname}")

            # Augment if enabled
            if use_augmentation:
                for aug_audio in augment(audio):
                    aug_feat = extract_features(aug_audio)
                    if not (np.any(np.isnan(aug_feat)) or
                            np.any(np.isinf(aug_feat))):
                        X.append(aug_feat)
                        y_labels.append(label)

        class_counts[label] = ok_count
        aug_total = len([l for l in y_labels if l == label])
        print(f"    → {ok_count} originals  +  {aug_total - ok_count} augmented"
              f"  =  {aug_total} samples")

    print(f"\n  Total samples : {len(X)}")
    print(f"  Class balance : {Counter(y_labels)}")

    # ── Warn if severely imbalanced ─────────────────────────────
    counts = Counter(y_labels)
    min_c  = min(counts.values())
    max_c  = max(counts.values())
    if max_c > min_c * 2:
        print(f"\n  [WARN] Class imbalance detected ({min_c} vs {max_c})")
        print(f"         Using class_weight='balanced' to compensate.")

    return np.array(X), np.array(y_labels)

# ================================================================
# TRAIN MULTIPLE MODELS, PICK THE BEST
# ================================================================
def train_models(X_train, X_test, y_train, y_test, le):
    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight='balanced',   # handles imbalanced data
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,          # needed for predict_proba
            random_state=42
        ),
    }

    best_name  = None
    best_score = 0.0
    best_model = None
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'═'*52}")
    print("  Training & cross-validating models...")
    print(f"{'═'*52}")

    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_train, y_train,
                                  cv=cv, scoring='f1_macro', n_jobs=-1)
        mean_f1 = scores.mean()
        print(f"  {name:<20} CV F1 = {mean_f1:.3f} ± {scores.std():.3f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_name  = name
            best_model = clf

    print(f"\n  ★ Best model: {best_name}  (CV F1 = {best_score:.3f})")

    # Final fit on full training set
    best_model.fit(X_train, y_train)

    # ── Evaluation on held-out test set ──────────────────────────
    y_pred = best_model.predict(X_test)
    y_enc  = le.transform(y_test)
    p_enc  = le.transform(y_pred)

    print(f"\n{'═'*52}")
    print("  Test-set evaluation")
    print(f"{'═'*52}")
    print(classification_report(y_test, y_pred,
                                  target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    print("  Confusion matrix:")
    print(f"                Pred NORMAL  Pred MURMUR")
    for i, row_label in enumerate(le.classes_):
        print(f"  True {row_label:<8}  {cm[i][0]:>9}    {cm[i][1]:>9}")

    return best_model, best_score

# ================================================================
# MAIN
# ================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   CardioScope — Model Trainer  v2.0              ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║   Target sample rate : {TARGET_SR} Hz (matches ESP32)    ║")
    print(f"║   Clip length        : {RECORD_SECONDS}s / {TARGET_SAMPLES} samples          ║")
    print(f"║   Feature vector     : 95 values                  ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Sanity check ─────────────────────────────────────────────
    if not os.path.isdir(DATA_DIR):
        print(f"\n[ERROR] 'dataset/' folder not found.")
        print(f"  Expected this structure:")
        print(f"    dataset/")
        print(f"      Normal/   <- .wav files of normal heart sounds")
        print(f"      Murmur/   <- .wav files of murmur heart sounds")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    X, y_labels = load_dataset(use_augmentation=True)

    if len(X) < 10:
        print(f"\n[ERROR] Only {len(X)} samples loaded — need at least 10.")
        print("        Add more .wav files to data/normal/ and data/murmur/")
        sys.exit(1)

    # ── Encode labels ─────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y_labels)

    # ── Scale features ────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split (stratified) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_labels,
        test_size=0.2,
        stratify=y_labels,
        random_state=42
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────────
    best_model, best_f1 = train_models(X_train, X_test, y_train, y_test, le)

    # ── Save ─────────────────────────────────────────────────────
    model_path  = os.path.join(MODELS_DIR, 'stethoscope_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    le_path     = os.path.join(MODELS_DIR, 'label_encoder.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler,     scaler_path)
    joblib.dump(le,         le_path)

    print(f"\n{'═'*52}")
    print(f"  Models saved to  {MODELS_DIR}/")
    print(f"    stethoscope_model.pkl")
    print(f"    scaler.pkl")
    print(f"    label_encoder.pkl")
    print(f"  Best CV F1 score : {best_f1:.3f}")
    print(f"{'═'*52}")

    if best_f1 < 0.70:
        print(f"\n  [WARN] F1 < 0.70 — model may not be reliable.")
        print(f"  Tips to improve:")
        print(f"    • Add more murmur recordings (aim for ≥30 per class)")
        print(f"    • Use PhysioNet 2016 Challenge dataset")
        print(f"    • Ensure recordings are actual heart sounds (not breath)")
    else:
        print(f"\n  ✓ Model looks good! Restart local_prediction_server.py")
        print(f"    to load the new model.")

    print()

if __name__ == '__main__':
    main()

