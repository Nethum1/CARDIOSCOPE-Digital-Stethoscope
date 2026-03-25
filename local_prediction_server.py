# ================================================================
# local_prediction_server.py  — CardioScope API Server
#
# THIS FILE IS API ONLY — no dashboard serving.
# Open dashboard.html directly in your browser.
# The dashboard connects here via fetch() calls.
#
# Pipeline:
#   ESP32 → POST /predict (raw int16 bytes)
#   Flask → librosa features → ML model → JSON response
#   Dashboard (dashboard.html) → polls GET /latest every 2s
#
# Matches ESP32 firmware:
#   Sample rate  : 800 SPS  (from X-Sample-Rate header)
#   Record time  : 5 seconds
#   Total samples: 4000
#   Payload      : 8000 bytes
#   ADC gain     : GAIN_TWO (MAX9814 @ 3.3V)
#
# Run  : python local_prediction_server.py
# Test : http://localhost:5000/test   (no ESP32 needed)
# Status: http://localhost:5000/status
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile, os
from datetime import datetime
from collections import deque
import threading

app = Flask(__name__)

# ── CORS: allow dashboard.html opened as file:// or any origin ──
CORS(app, resources={r"/*": {"origins": "*"}})

# ================================================================
# LOAD MODEL
# ================================================================
try:
    model  = joblib.load('models/stethoscope_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le     = joblib.load('models/label_encoder.pkl')
    MODEL_LOADED = True
    print("[OK] ML model loaded!")
    print(f"[OK] Classes: {le.classes_.tolist()}")
except Exception as e:
    MODEL_LOADED = False
    print(f"[WARN] No model found ({e})")
    print("[WARN] Running in DEMO mode — run train_local.py first")

# ================================================================
# CONSTANTS  — must match ESP32 firmware
# ================================================================
DEFAULT_SAMPLE_RATE = 800     # ADS1115 @ 800 SPS
RECORD_SECONDS      = 5       # 5 second recording
DEFAULT_SAMPLES     = DEFAULT_SAMPLE_RATE * RECORD_SECONDS  # 4000

CLASSES  = ['normal', 'murmur']
MESSAGES = {
    'normal'    : 'Heart sounds appear NORMAL. No anomalies detected.',
    'murmur'    : 'Possible heart MURMUR detected. Consult a cardiologist.'
}

# Thread-safe in-memory storage
history     = deque(maxlen=50)
latest_data = {}
lock        = threading.Lock()

# ================================================================
# FEATURE EXTRACTION
# n_fft=512 is safe for 800 SPS × 5s = 4000 samples
# IDENTICAL to train_local.py — must stay in sync
# ================================================================
def extract_all(audio_float, sample_rate):
    y  = audio_float
    sr = sample_rate

    n_fft      = min(512, len(y))
    hop_length = n_fft // 4

    # MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40,
        n_fft=n_fft, hop_length=hop_length
    )

    # Supporting features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_c = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms    = librosa.feature.rms(y=y, hop_length=hop_length)

    # Feature vector for ML model (~95 values)
    features = np.concatenate([
        np.mean(mfcc,   axis=1),   # 40
        np.std (mfcc,   axis=1),   # 40
        np.mean(chroma, axis=1),   # 12
        [np.mean(spec_c)],          #  1
        [np.mean(zcr)],             #  1
        [np.mean(rms)],             #  1
    ])

    # Waveform — downsample to 200 points for dashboard chart
    idx       = np.linspace(0, len(y)-1, 200, dtype=int)
    wave_down = y[idx].tolist()

    # MFCC matrix — downsample time axis to 60 cols for heatmap
    t_cols  = mfcc.shape[1]
    t_idx   = np.linspace(0, t_cols-1, min(60, t_cols), dtype=int)
    mfcc_60 = mfcc[:, t_idx]
    mn, mx  = mfcc_60.min(), mfcc_60.max()
    mfcc_norm = ((mfcc_60 - mn) / (mx - mn)).tolist() if mx > mn \
                else np.zeros_like(mfcc_60).tolist()

    # Scalar quality metrics
    rms_db    = float(20 * np.log10(np.mean(rms) + 1e-9))
    zcr_mean  = float(np.mean(zcr))
    spec_mean = float(np.mean(spec_c))

    return {
        'features'     : features,
        'mfcc_matrix'  : mfcc_norm,     # list[40][60]
        'waveform_down': wave_down,     # list[200]
        'rms_db'       : round(rms_db,   2),
        'zcr_mean'     : round(zcr_mean, 5),
        'spec_centroid': round(spec_mean,1),
    }

# ================================================================
# PREDICTION CORE
# ================================================================
def run_prediction(audio_float, sample_rate):
    extracted  = extract_all(audio_float, sample_rate)
    features   = extracted['features']

    if MODEL_LOADED:
        feat_scaled = scaler.transform([features])
        proba       = model.predict_proba(feat_scaled)[0]
        pred_idx    = int(np.argmax(proba))
        label       = le.classes_[pred_idx]
        confidence  = float(proba[pred_idx])
        all_proba   = {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    else:
        # Demo mode
        import random
        label      = random.choice(CLASSES)
        confidence = round(random.uniform(0.62, 0.95), 4)
        rest       = (1.0 - confidence) / (len(CLASSES) - 1)
        all_proba  = {
            c: (confidence if c == label
                else round(rest + random.uniform(-0.04, 0.04), 4))
            for c in CLASSES
        }

    now = datetime.now()
    return {
        # Prediction fields
        'prediction'        : label,
        'confidence'        : confidence,
        'all_probabilities' : all_proba,
        'message'           : MESSAGES.get(label, ''),
        'model_loaded'      : MODEL_LOADED,
        'timestamp'         : now.strftime('%H:%M:%S'),
        'datetime'          : now.strftime('%Y-%m-%d %H:%M:%S'),

        # Real signal data — sent to dashboard for charts
        'waveform'          : extracted['waveform_down'],   # 200 pts
        'mfcc_matrix'       : extracted['mfcc_matrix'],     # 40×60

        # Audio quality metrics — shown in dashboard stats bar
        'rms_db'            : extracted['rms_db'],
        'zcr'               : extracted['zcr_mean'],
        'spectral_centroid' : extracted['spec_centroid'],

        # Audio metadata — dashboard uses these for labels
        'audio_samples'     : len(audio_float),
        'sample_rate'       : sample_rate,
        'duration_sec'      : round(len(audio_float) / sample_rate, 2),
        'mic_type'          : 'MAX9814-ADS1115',
        'adc_gain'          : 'GAIN_TWO',
    }

# ================================================================
# API ROUTES
# ================================================================

# ── /predict ─────────────────────────────────────────────────────
# ESP32 POSTs raw int16 audio bytes here
@app.route('/predict', methods=['POST'])
def predict():
    raw = request.get_data()

    # Read metadata headers sent by ESP32
    sample_rate  = int(request.headers.get('X-Sample-Rate',  DEFAULT_SAMPLE_RATE))
    record_secs  = int(request.headers.get('X-Record-Secs',  RECORD_SECONDS))
    mic_type     = request.headers.get('X-Mic-Type',  'MAX9814-ADS1115-3V3')
    adc_gain     = request.headers.get('X-ADC-Gain',  'GAIN_TWO')

    expected = sample_rate * record_secs * 2   # int16 = 2 bytes each
    min_ok   = sample_rate * 1 * 2             # at least 1 second

    print(f"\n{'─'*48}")
    print(f"[ESP32] Received  : {len(raw):,} bytes")
    print(f"[ESP32] Expected  : {expected:,} bytes  ({sample_rate}×{record_secs}s×2)")
    print(f"[ESP32] Mic       : {mic_type}  |  Gain: {adc_gain}")

    if len(raw) < min_ok:
        msg = f"Too short: {len(raw)} bytes (need ≥ {min_ok})"
        print(f"[ERROR] {msg}")
        return jsonify({'error': msg}), 400

    # Convert raw int16 bytes → float32  (-1.0 … +1.0)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[PROC]  Samples   : {len(audio)}  →  {len(audio)/sample_rate:.1f}s")
    print(f"[PROC]  Range     : {audio.min():.4f}  to  {audio.max():.4f}")
    print(f"[PROC]  RMS level : {np.sqrt(np.mean(audio**2)):.5f}")

    # Save to temp WAV for librosa
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = tmp.name

    try:
        result = run_prediction(audio, sample_rate)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Store latest + history
    with lock:
        latest_data.clear()
        latest_data.update(result)
        compact = {k: v for k, v in result.items()
                   if k not in ('waveform', 'mfcc_matrix')}
        history.appendleft(compact)

    print(f"[PRED]  {result['prediction'].upper():<13}"
          f"conf={result['confidence']:.1%}  "
          f"@ {result['timestamp']}")
    print(f"{'─'*48}")

    return jsonify(result)


# ── /latest ──────────────────────────────────────────────────────
# Dashboard polls this every 2 seconds
# Returns full prediction including waveform + mfcc_matrix
@app.route('/latest', methods=['GET'])
def latest():
    with lock:
        if latest_data:
            return jsonify(dict(latest_data))
    return jsonify({'empty': True})


# ── /history ─────────────────────────────────────────────────────
# Compact history — no large arrays
@app.route('/history', methods=['GET'])
def get_history():
    with lock:
        return jsonify(list(history))


# ── /status ──────────────────────────────────────────────────────
# Dashboard checks this to know if server + model are ready
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'running'           : True,
        'model_loaded'      : MODEL_LOADED,
        'total_predictions' : len(history),
        'sample_rate'       : DEFAULT_SAMPLE_RATE,
        'record_seconds'    : RECORD_SECONDS,
        'total_samples'     : DEFAULT_SAMPLES,
        'mic_type'          : 'MAX9814-ADS1115',
        'adc_gain'          : 'GAIN_TWO',
    })


# ── /test ────────────────────────────────────────────────────────
# Injects a synthetic heartbeat signal — test without ESP32
@app.route('/test', methods=['GET'])
def test_predict():
    sr  = DEFAULT_SAMPLE_RATE   # 800
    dur = RECORD_SECONDS        # 5
    t   = np.linspace(0, dur, sr * dur)

    # Synthetic heartbeat at ~80 BPM (1.33 Hz)
    audio = (
        0.6 * np.sin(2 * np.pi * 1.33 * t) * np.exp(-5 * np.mod(t, 0.75)) +
        0.3 * np.sin(2 * np.pi * 80 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.7

    result = run_prediction(audio, sr)

    with lock:
        latest_data.clear()
        latest_data.update(result)
        compact = {k: v for k, v in result.items()
                   if k not in ('waveform', 'mfcc_matrix')}
        history.appendleft(compact)

    print(f"[TEST] Injected: {result['prediction']} "
          f"({result['confidence']:.1%})")

    return jsonify({
        'message'   : 'Test prediction injected — check dashboard',
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'timestamp' : result['timestamp'],
    })

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      CardioScope — Local Prediction API           ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  API Endpoints:                                    ║")
    print("║    ESP32  → POST http://<PC_IP>:5000/predict      ║")
    print("║    Test   → GET  http://localhost:5000/test        ║")
    print("║    Latest → GET  http://localhost:5000/latest      ║")
    print("║    Status → GET  http://localhost:5000/status      ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Dashboard:                                        ║")
    print("║    Open dashboard.html directly in browser         ║")
    print("║    It connects to localhost:5000 automatically     ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Sample Rate : {DEFAULT_SAMPLE_RATE} SPS                         ║")
    print(f"║  Record Time : {RECORD_SECONDS} seconds                      ║")
    print(f"║  Samples     : {DEFAULT_SAMPLES} per recording              ║")
    print(f"║  Mic         : MAX9814 + ADS1115 (3.3V)           ║")
    print(f"║  Model       : {'LOADED ✓' if MODEL_LOADED else 'DEMO MODE (train first)'}                    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # host='0.0.0.0' so ESP32 on same WiFi can reach it
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
