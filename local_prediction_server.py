# ================================================================
# local_prediction_server.py  — CardioScope API Server  v2.3
#
# HOTSPOT FIX NOTES:
#   This file is unchanged from v2.2 — it already binds to
#   host='0.0.0.0' which is correct.
#
#   The real fix is on the ESP32 side (PC_IP value) and in
#   Windows Firewall. But this version prints your correct
#   hotspot IP clearly on startup so you can copy it into
#   the ESP32 firmware without running ipconfig separately.
#
# Run  : python local_prediction_server.py
# Test : http://localhost:5000/test
# Status: http://localhost:5000/status
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile, os, socket
from datetime import datetime
from collections import deque
import threading

app = Flask(__name__)
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
# CONSTANTS
# ================================================================
DEFAULT_SAMPLE_RATE = 800
RECORD_SECONDS      = 5
DEFAULT_SAMPLES     = DEFAULT_SAMPLE_RATE * RECORD_SECONDS

CLASSES  = ['normal', 'murmur']
MESSAGES = {
    'normal' : 'Heart sounds appear NORMAL. No anomalies detected.',
    'murmur' : 'Possible heart MURMUR detected. Consult a cardiologist.'
}

history     = deque(maxlen=50)
latest_data = {}
lock        = threading.Lock()

# ================================================================
# HELPER — find the IP on the active Wi-Fi / hotspot interface
# ================================================================
def get_hotspot_ip():
    """
    Returns the machine's IP on the active non-loopback network.
    Works on Windows, macOS, Linux.
    Prefers IPs in common hotspot ranges (192.168.x.x, 172.x.x.x).
    """
    candidates = []
    try:
        # Trick: connect a UDP socket (no data sent) to discover local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        candidates.append(ip)
    except Exception:
        pass

    # Fallback: enumerate all interfaces
    try:
        hostname = socket.gethostname()
        all_ips  = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for info in all_ips:
            ip = info[4][0]
            if not ip.startswith('127.'):
                candidates.append(ip)
    except Exception:
        pass

    # Prefer hotspot-range IPs
    for ip in candidates:
        if ip.startswith('192.168.') or ip.startswith('172.') or ip.startswith('10.'):
            return ip

    return candidates[0] if candidates else '(unknown — run ipconfig)'

# ================================================================
# FEATURE EXTRACTION (unchanged)
# ================================================================
def extract_all(audio_float, sample_rate):
    y  = audio_float
    sr = sample_rate

    n_fft      = min(512, len(y))
    hop_length = n_fft // 4

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                   n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_c = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms    = librosa.feature.rms(y=y, hop_length=hop_length)

    features = np.concatenate([
        np.mean(mfcc,   axis=1),
        np.std (mfcc,   axis=1),
        np.mean(chroma, axis=1),
        [np.mean(spec_c)],
        [np.mean(zcr)],
        [np.mean(rms)],
    ])

    idx       = np.linspace(0, len(y)-1, 200, dtype=int)
    wave_down = y[idx].tolist()

    t_cols  = mfcc.shape[1]
    t_idx   = np.linspace(0, t_cols-1, min(60, t_cols), dtype=int)
    mfcc_60 = mfcc[:, t_idx]
    mn, mx  = mfcc_60.min(), mfcc_60.max()
    mfcc_norm = ((mfcc_60 - mn) / (mx - mn)).tolist() if mx > mn \
                else np.zeros_like(mfcc_60).tolist()

    rms_db    = float(20 * np.log10(np.mean(rms) + 1e-9))
    zcr_mean  = float(np.mean(zcr))
    spec_mean = float(np.mean(spec_c))

    return {
        'features'     : features,
        'mfcc_matrix'  : mfcc_norm,
        'waveform_down': wave_down,
        'rms_db'       : round(rms_db,   2),
        'zcr_mean'     : round(zcr_mean, 5),
        'spec_centroid': round(spec_mean, 1),
    }

# ================================================================
# PREDICTION CORE (unchanged)
# ================================================================
def run_prediction(audio_float, sample_rate):
    extracted = extract_all(audio_float, sample_rate)
    features  = extracted['features']

    if MODEL_LOADED:
        feat_scaled = scaler.transform([features])
        proba       = model.predict_proba(feat_scaled)[0]
        pred_idx    = int(np.argmax(proba))
        label       = le.classes_[pred_idx]
        confidence  = float(proba[pred_idx])
        all_proba   = {le.classes_[i]: round(float(p), 4)
                       for i, p in enumerate(proba)}
    else:
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
        'prediction'        : label,
        'confidence'        : confidence,
        'all_probabilities' : all_proba,
        'message'           : MESSAGES.get(label, ''),
        'model_loaded'      : MODEL_LOADED,
        'timestamp'         : now.strftime('%H:%M:%S'),
        'datetime'          : now.strftime('%Y-%m-%d %H:%M:%S'),
        'waveform'          : extracted['waveform_down'],
        'mfcc_matrix'       : extracted['mfcc_matrix'],
        'rms_db'            : extracted['rms_db'],
        'zcr'               : extracted['zcr_mean'],
        'spectral_centroid' : extracted['spec_centroid'],
        'audio_samples'     : len(audio_float),
        'sample_rate'       : sample_rate,
        'duration_sec'      : round(len(audio_float) / sample_rate, 2),
        'mic_type'          : 'MAX9814-ADS1115',
        'adc_gain'          : 'GAIN_TWO',
    }

# ================================================================
# API ROUTES
# ================================================================

@app.route('/predict', methods=['POST'])
def predict():
    raw = request.get_data()

    sample_rate = int(request.headers.get('X-Sample-Rate', DEFAULT_SAMPLE_RATE))
    record_secs = int(request.headers.get('X-Record-Secs', RECORD_SECONDS))
    mic_type    = request.headers.get('X-Mic-Type',  'MAX9814-ADS1115-3V3')
    adc_gain    = request.headers.get('X-ADC-Gain',  'GAIN_TWO')

    expected = sample_rate * record_secs * 2
    min_ok   = sample_rate * 1 * 2

    print(f"\n{'─'*48}")
    print(f"[ESP32] Received  : {len(raw):,} bytes")
    print(f"[ESP32] Expected  : {expected:,} bytes  ({sample_rate}×{record_secs}s×2)")
    print(f"[ESP32] Mic       : {mic_type}  |  Gain: {adc_gain}")

    if len(raw) < min_ok:
        msg = f"Too short: {len(raw)} bytes (need ≥ {min_ok})"
        return jsonify({'error': msg}), 400

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[PROC]  Samples   : {len(audio)}  →  {len(audio)/sample_rate:.1f}s")
    print(f"[PROC]  Range     : {audio.min():.4f}  to  {audio.max():.4f}")
    print(f"[PROC]  RMS level : {np.sqrt(np.mean(audio**2)):.5f}")

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

    with lock:
        latest_data.clear()
        latest_data.update(result)
        compact = {k: v for k, v in result.items()
                   if k not in ('waveform', 'mfcc_matrix')}
        history.appendleft(compact)

    print(f"[PRED]  {result['prediction'].upper():<13}"
          f"conf={result['confidence']:.1%}  @ {result['timestamp']}")
    print(f"{'─'*48}")

    return jsonify(result)


@app.route('/latest', methods=['GET'])
def latest():
    with lock:
        if latest_data:
            return jsonify(dict(latest_data))
    return jsonify({'empty': True})


@app.route('/history', methods=['GET'])
def get_history():
    with lock:
        return jsonify(list(history))


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


@app.route('/test', methods=['GET'])
def test_predict():
    sr  = DEFAULT_SAMPLE_RATE
    dur = RECORD_SECONDS
    t   = np.linspace(0, dur, sr * dur)

    audio = (
        0.6 * np.sin(2 * np.pi * 1.33 * t) * np.exp(-5 * np.mod(t, 0.75)) +
        0.3 * np.sin(2 * np.pi * 80  * t) +
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

    print(f"[TEST] Injected: {result['prediction']} ({result['confidence']:.1%})")

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

    # ── Auto-detect hotspot IP and print it prominently ──────────
    hotspot_ip = get_hotspot_ip()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      CardioScope — Local Prediction API  v2.3    ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║                                                    ║")
    print(f"║  *** USE THIS IP IN ESP32 FIRMWARE ***             ║")
    print(f"║      PC_IP = \"{hotspot_ip}\"".ljust(52) + "║")
    print("║                                                    ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  API Endpoints:                                    ║")
    print(f"║    ESP32  → POST http://{hotspot_ip}:5000/predict".ljust(52) + "║")
    print(f"║    Test   → GET  http://localhost:5000/test        ║")
    print(f"║    Status → GET  http://localhost:5000/status      ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  If ESP32 still shows 'Server offline':            ║")
    print("║  1. Check IP above matches ipconfig Wi-Fi adapter  ║")
    print("║  2. Allow port 5000 in Windows Firewall (TCP in)   ║")
    print("║  3. Both devices on SAME hotspot network?          ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Model: {'LOADED ✓' if MODEL_LOADED else 'DEMO MODE — run train_local.py'}".ljust(52) + "║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
