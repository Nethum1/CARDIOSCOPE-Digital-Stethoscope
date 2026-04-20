# 🩺CARDIOSCOPE-(Digital Stethoscope)

<br>
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
      <ul>
        <li><a href="#technologies-used">Technologies Used</a></li>
      </ul>
    </li>
    <li><a href="#objectives">Objectives</a></li>
    <li><a href="#esp32">ESP32</a></li>
    <li>
      <a href="#system-architecture">System Architecture</a>
      <ul>
        <li><a href="#pipeline">Complete Pipeline</a></li>
      </ul>
    </li>
    <li>
      <a href="#hardware">Hardware</a>
      <ul>
        <li><a href="#heart-sounds">Heart & Lung Sounds</a></li>
      </ul>
    </li>
    <li>
      <a href="#ml-model">ML Model</a>
      <ul>
        <li><a href="#feature-extraction">Feature Extraction (MFCC)</a></li>
        <li><a href="#random-forest">Random Forest Classifier</a></li>
        <li><a href="#cnn-model">CNN Deep Learning Model</a></li>
      </ul>
    </li>
    <li>
      <a href="#digital-signal-processing-pipeline">Digial Signal Processing Pipeline</a>
    </li>
    <li><a href="#dashboard">Real-Time Dashboard</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#setup-guide">Setup Guide</a></li>
  </ol>
</details>

## Introduction
This project builds a fully functional **Digital Stethoscope** system using an ESP32 microcontroller and MAX9814 I2S digital microphone. The system captures heart and lung sounds, streams the raw audio wirelessly over Wi-Fi to a local PC, runs a Machine Learning model to classify the sounds, and delivers **real-time diagnostic predictions** through a live web dashboard — all without any cloud dependency.

The key advantages of this digital approach over a traditional mechanical stethoscope are:
- Real-time audio capture using a high-quality **I2S digital microphone** (MAX9814)
- Wireless transmission to a local PC over the **same Wi-Fi network**
- **Machine Learning classification** —> Normal, Murmur (In my first step)
- Live web dashboard with **real waveform**, probability charts
- **Patient record management** with PDF report export
- Fully local — **no cloud required**, no internet dependency
<hr>

## Technologies Used
We have used the following technologies and software for our project

<ul>
    <li>
        <a href="https://www.python.org/" target ="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" alt="python "
        width = "50"
        height = "50"></a>
    </li>
    <li>
        <a href="https://www.tensorflow.org/" target ="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" alt="TensorFlow "
        width = "50"
        height ="50"></a>
    </li>
    <li>
    <a href="https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html" target ="_blank"><img src="https://github.com/gagan20003/Digital-Stethoscope/blob/main/resources/esp%20logo.jpeg" alt="ESP32"
        width = "50"
        height = "50"></a>
    </li>
    <li>
    <a href="https://librosa.org/" target="_blank">
      <img src="https://librosa.org/doc/latest/_static/librosa_logo_text.png"
      alt="librosa" width="100" height="40">
    </a>&nbsp; Librosa (Audio Feature Extraction)
  </li>
  <li>
    <a href="https://code.visualstudio.com/" target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png"
      alt="VS Code" width="50" height="50">
    </a>&nbsp; Visual Studio Code + PlatformIO IDE
  </li>
  <li>
    <a href="https://scikit-learn.org/" target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"
      alt="scikit-learn" width="80" height="50">
    </a>&nbsp; Scikit-Learn (Random Forest)
  </li>
</ul>
<hr>

## Objectives 
The main goals have set out for this project are 
- Make a microphone amplification module that can detect frequencies from 20Hz to 20kHz 
- Transmit the sound clip to an associated laptop wirelessly 
- Be able to detect the disease by using an associated Deep Learning model
<hr>

<h2>ESP32</h2>
<img src="https://cdn.shopify.com/s/files/1/0609/6011/2892/files/doc-esp32-pinout-reference-wroom-devkit.png?width=1384" alt="ESP32 Pinout"
  width="700"
  height="500"
  align="middle"

<p>The microcontroller that we are using in our project is the ESP32 WROOM - 32.
The ESP32 is a high-performance microcontroller board that is widely used in Internet of Things (IoT) applications.
Espressif Systems manufactures it, and it is the successor to the ESP8266. The ESP32 is built around a dual-core 
Tensilica LX6 processor that operates at up to 240 MHz, providing plenty of processing power for IoT applications.
</p>

The ESP32 microcontroller board has the following key features:
<ol>
<li>
    <em>Wi-Fi and Bluetooth connectivity:</em> The ESP32 includes Wi-Fi and Bluetooth connectivity, making it simple to connect to the internet and other devices.
</li>
<li>
    <em> Low power consumption:</em> The ESP32 is energy-efficient, with multiple power modes that allow it to operate on very little power.
</li>
<li>
    <em> Large Memory:</em> The ESP32 comes up with up to 520 KB of SRAM and 4MB of flash memory, which is enough to run complex applications.
</li>
<li>
    <em>GPIO pins:</em> The ESP32 has a large number of GPIO pins, which makes it easy to interface with other devices and sensors.
</li>
<li>
    <em>Development tools:</em> There are many development tools available for it including Arduino IDE, ESP-IDF, and MicroPython.
</li>
</ol>
<p>
Overall, the ESP32 is a powerful and versatile microcontroller board that is well-suited for a wide range of IoT applications. Its built-in Wi-Fi and Bluetooth capabilities, low power consumption, and large memory make it a popular choice among developers.

In this project, the esp32 is used to transmit the data collected from the microphone to the ML model over the in-built Wi-Fi.
</p> <hr>

## System Architecture

### Pipeline

The complete data flow of the system:

```
┌─────────────────┐     I2S Digital Audio      ┌──────────────────────┐
│  MAX9814 Mic    │ ─────────────────────────> │   ESP32 WROOM - 32   │
│     (8 bit)     │                            │   (PlatformIO)       │
└─────────────────┘                            └──────────┬───────────┘
                                                          │
                                              HTTP POST /predict
                                              Same Wi-Fi Network
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LOCAL PC — Flask Server (Python)                  │
│                                                                      │
│  Raw bytes → float32 audio → librosa feature extraction              │
│       ↓                                                              │
│  • MFCC (40 coeff × 60 frames)   — for heatmap + model input         │
│  • Waveform (200 downsampled pts) — for waveform chart               │
│  • RMS dB, ZCR, Spectral Centroid — for signal metrics               │
│       ↓                                                              │
│  Random Forest Classifier → prediction + all class probabilities     │
│       ↓                                                              │
│  /latest endpoint stores full result                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                   Dashboard polls /latest every 2s
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (dashboard.html)                    │
│                                                                      │
│  ● Real waveform chart          ● MFCC heatmap (40×60)               │
│  ● Class probability bars       ● Confidence trend line              │
│  ● Session distribution donut   ● Signal metrics (RMS, SC, ZCR)      │
│  ● Patient record form          ● PDF report export                  │
└──────────────────────────────────────────────────────────────────────┘
```

<hr>

## Hardware

### Components

| Component | Specification |
|---|---|
| **Microcontroller** | ESP32-WROOM-32 |
| **Microphone** | MAX9814 I2S Digital Microphone (8-bit) |
| **Stethoscope** | Acoustic stethoscope —> mic inserted into chest piece |
| **Switch** | Power ON and OFF|
| **Push Button** | Momentary button —> manual recording trigger |
| **Power** | USB 5V or 3.7V Li-Po battery + TP4056 charger |
| **Resistors** | 10KΩ × 1 |
| **Dotboard** | soldering board |
| **Wires** |20cm |

<hr>

## Circuit Diagram
<img width="476" height="347" alt="circuit for cardioscope" src="https://github.com/user-attachments/assets/23de4e5f-b250-4a58-b181-496411754ae4" />

<hr>

## Heart & Lung Sounds

Heart and lung sounds carry important diagnostic information:

**Heart Sounds:**
- **Normal (S1/S2):** The classic "lub-dub" — closure of mitral/tricuspid valves (S1) and aortic/pulmonic valves (S2), frequency range 20–600 Hz
- **Murmur:** Abnormal whooshing caused by turbulent blood flow — detectable as increased high-frequency energy in the signal
- **Arrhythmia:** Irregular lub-dub timing — detectable via irregular zero-crossing patterns and RMS energy variance
- **Extra Sound (S3/S4):** Additional heart sounds indicating heart failure or stiffness

**Lung Sounds:**
- **Bronchial:** Loud, high-pitched sounds over the trachea and larger airways
- **Vesicular:** Softer sounds over smaller airways
- **Crackles (Rales):** Discontinuous sounds caused by fluid-filled or collapsed airways reopening
- **Wheezes:** Continuous high-pitched sounds from narrowed airways (asthma, COPD)

The MAX9814 captures sounds in the 20Hz–20kHz frequency response with automatic gain control (AGC). Heart sounds (20–600Hz) and lung sounds (up to 2000Hz) both fall well within the captured band. The ESP32 DMA buffers 3 seconds of audio (48,000 samples) before transmitting.

<hr>

## ML Model

### Feature Extraction (MFCC)

Raw audio cannot be fed directly into an ML model. The Flask server uses **librosa** to extract a feature vector of ~95 values from every 3-second recording:

| Feature | Count | Description |
|---|---|---|
| MFCC mean | 40 | Mean of 40 Mel-Frequency Cepstral Coefficients over time |
| MFCC std | 40 | Standard deviation — captures how the sound varies |
| Chroma mean | 12 | Pitch-class energy distribution |
| Spectral Centroid | 1 | "Brightness" of the sound (Hz) |
| Zero Crossing Rate | 1 | How often the signal crosses zero — relates to noisiness |
| RMS Energy | 1 | Loudness of the recording (dB) |

The **MFCC matrix (40×60)** is also sent back to the dashboard for the real-time heatmap visualization.

**Why MFCC?** MFCCs mimic how the human auditory system perceives sound — lower coefficients capture broad spectral shape (useful for murmur detection) while higher coefficients capture fine timbral texture (useful for distinguishing arrhythmia patterns).

### Random Forest Classifier

The **primary model** used for real-time prediction is a Random Forest Classifier:

```python
model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                random_state=42, n_jobs=-1)
```

**How it works:**
- Trains 200 decision trees, each learning different patterns from the ~95 audio features
- For each prediction, all 200 trees vote — the majority class wins
- Outputs both the predicted class **and** probability for each class
- Trained in seconds, runs predictions in <100ms

**Classes:** `normal` | `murmur` | `arrhythmia` | `extrasound`

**Why Random Forest over Deep Learning?**

| | Random Forest | CNN/LSTM |
|---|---|---|
| Training data needed | 100+ samples | 1000+ samples |
| Training time | Seconds | Minutes–hours |
| Accuracy (small dataset) | 85–92% | Needs more data |
| GPU required | No | Recommended |
| Explainability | High | Low |

### CNN Deep Learning Model

A **1D Convolutional Neural Network** is also implemented as an advanced alternative using TensorFlow/Keras:

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (time_frames, 40) | — |
| Conv1D (64 filters) | (time, 64) | 7,744 |
| BatchNormalization | (time, 64) | 256 |
| MaxPooling1D | (time/2, 64) | — |
| Dropout (0.3) | (time/2, 64) | — |
| Conv1D (128 filters) | (time/2, 128) | 24,704 |
| BatchNormalization | (time/2, 128) | 512 |
| MaxPooling1D | (time/4, 128) | — |
| Dropout (0.3) | (time/4, 128) | — |
| Conv1D (256 filters) | (time/4, 256) | 98,560 |
| GlobalAveragePooling1D | (256,) | — |
| Dense (128) | (128,) | 32,896 |
| Dropout (0.4) | (128,) | — |
| Dense (4, softmax) | (4,) | 516 |

The CNN treats the MFCC matrix as a temporal sequence and uses convolutional filters to detect local patterns like the characteristic S1–S2 timing of a normal heartbeat or the irregular bursts of an arrhythmia.

<hr>

## DSP Signal Processing Pipeline

> **Digital signal processing applied on-device (ESP32) to clean raw heart sound recordings before ML classification.**



### Why DSP?

The MAX9814 microphone captures everything — not just heart sounds. Raw recordings contain several layers of interference that directly hurt ML model accuracy:

| Noise Source | Frequency | Origin |
|---|---|---|
| DC offset / baseline drift | 0 – 5 Hz | Capacitor coupling, temperature |
| Breathing artifacts | 5 – 15 Hz | Patient movement during recording |
| Power line hum | **50 Hz** | Mains electricity (strongest indoor noise) |
| High-frequency hiss | 400 – 800 Hz | ADC quantisation, amplifier noise |
| Useful heart sounds | **20 – 300 Hz** | S1, S2 (lub-dub), murmurs |

Without filtering, the ML model receives a noisy signal where interference can be as strong as or stronger than the actual heart sounds. The DSP pipeline removes everything outside the 20–300 Hz cardiac band before the signal ever leaves the ESP32.



### Pipeline Overview

Processing runs **after** recording and **before** HTTP transmission to the Flask server. This two-stage separation is intentional — the recording loop runs with microsecond-precision timing (1250 µs between samples at 800 SPS), and floating-point DSP math inside that loop would cause timing jitter and corrupt the waveform.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSP Processing Pipeline                      │
│                                                                 │
│  Raw ADC buffer (4000 × int16)                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │  1. High-Pass   │  @ 20 Hz  — removes DC + breathing rumble  │
│  │  Butterworth 2  │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  2. Notch       │  @ 50 Hz  — removes mains interference     │
│  │  IIR  Q = 35    │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  3. Low-Pass    │  @ 300 Hz — removes HF noise above cardiac │
│  │  Butterworth 2  │             band                           │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  4. Normalize   │  Peak → 80% of ±32767 full 16-bit range    │
│  │  Gain cap 20×   │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  Clean buffer → HTTP POST → Flask ML server                     │
└─────────────────────────────────────────────────────────────────┘
```


### Filter Design

All three filters use the **IIR Biquad (second-order section)** structure, sampled at **Fs = 800 Hz**.

#### Why IIR Biquad?

- Only **5 multiplications and 4 additions** per sample
- Only **4 state variables** (two delay taps for input, two for output)
- No significant RAM overhead on a 4000-sample buffer
- Hardware FPU on the ESP32 makes float arithmetic fast
- Industry-standard in embedded medical DSP (ECG, PCG, EEG)

#### Transfer Function

Every biquad stage follows this difference equation:

```
y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2]
                - a1·y[n-1] - a2·y[n-2]
```

Where `x[n]` is the current input sample and `y[n]` is the filtered output. The two `x` delay taps and two `y` delay taps form the filter's memory (state), which is reset to zero before each new recording.


### Stage 1 — High-Pass Filter @ 20 Hz

**Type:** Butterworth 2nd order
**Cutoff:** 20 Hz
**Purpose:** Removes everything below the cardiac band

```cpp
static const float HP_B0 =  0.89442f;
static const float HP_B1 = -1.78885f;
static const float HP_B2 =  0.89442f;
static const float HP_A1 = -1.77822f;
static const float HP_A2 =  0.79990f;
```

Heart sounds begin at approximately 20 Hz. Everything below is biological or electrical noise:

- **DC offset** — the MAX9814 output is centred at ~1.65V (half of 3.3V supply). The ADC midpoint subtraction in `recordAudio()` removes most of this, but residual drift remains.
- **Baseline wander** — slow movement of the stethoscope on the chest wall causes very low-frequency fluctuations.
- **Breathing artifacts** — respiratory movement modulates the baseline at 0.1–0.5 Hz (12–30 breaths/min).

Butterworth design was chosen for its maximally flat passband — it doesn't ripple or distort heart sounds near the 20 Hz cutoff.


### Stage 2 — Notch Filter @ 50 Hz

**Type:** IIR Notch (infinite impulse response)
**Centre frequency:** 50 Hz
**Quality factor (Q):** 35
**Purpose:** Removes mains power line interference

```cpp
static const float NOTCH_B0 =  0.99456f;
static const float NOTCH_B1 = -1.83817f;
static const float NOTCH_B2 =  0.99456f;
static const float NOTCH_A1 = -1.83817f;
static const float NOTCH_A2 =  0.98912f;
```

50 Hz mains hum is the **single strongest noise source** in any indoor medical device. It couples into the circuit through:

- Power supply ripple on the 3.3V rail
- Capacitive coupling from nearby cables and the human body
- Ground loops between the ESP32, ADS1115, and USB power

The high Q factor of 35 creates an extremely **narrow notch** — only frequencies within ±0.7 Hz of 50 Hz are attenuated. The rest of the spectrum, including the 45–55 Hz region that contains real cardiac sounds from patients with murmurs, is preserved.

> **Note for 60 Hz regions (Americas, Japan):** Change `NOTCH_B1` and `NOTCH_A1` to `-1.6180f` and recompute `NOTCH_A2 = 0.98912f` unchanged. The centre frequency shifts; Q and gain stay the same.


### Stage 3 — Low-Pass Filter @ 300 Hz

**Type:** Butterworth 2nd order
**Cutoff:** 300 Hz
**Purpose:** Removes high-frequency noise above the cardiac band

```cpp
static const float LP_B0 =  0.56901f;
static const float LP_B1 =  1.13803f;
static const float LP_B2 =  0.56901f;
static const float LP_A1 =  0.94281f;
static const float LP_A2 =  0.33333f;
```

The 300 Hz cutoff was chosen to capture the full cardiac frequency spectrum:

| Sound | Frequency content |
|---|---|
| S1 (first heart sound — "lub") | 10 – 140 Hz |
| S2 (second heart sound — "dub") | 10 – 150 Hz |
| S3, S4 (gallop sounds) | 20 – 70 Hz |
| Systolic murmurs | 100 – 300 Hz |
| Diastolic murmurs | 100 – 250 Hz |
| Aortic stenosis | up to 300 Hz |

At **Fs = 800 SPS**, the Nyquist limit is 400 Hz. A 300 Hz cutoff leaves a comfortable 100 Hz transition band before aliasing begins. Everything above 300 Hz (ADC quantisation noise, amplifier white noise, friction artefacts from the stethoscope tubing) is removed.


### Stage 4 — Amplitude Normalisation

**Purpose:** Scales every recording to a consistent amplitude for the ML model

```cpp
float gain = 26000.0f / (float)peak;
if (gain > 20.0f) gain = 20.0f;   // hard cap
```

Without normalisation, the ML model sees wildly different amplitude levels between recordings. The same patient pressing firmly vs lightly produces a 5–10× amplitude difference. Most classifiers — especially CNNs operating on spectrograms — are sensitive to absolute amplitude, not just spectral shape.

**Algorithm:**
1. Find the peak absolute sample value across all 4000 samples
2. Compute `gain = 26000 / peak` — this scales the peak to 80% of the int16 maximum (±32767)
3. Apply gain is capped at **20×** — if the signal is so quiet that 20× gain would be needed, the recording is likely just noise, and over-amplifying it would feed the ML model a magnified noise floor rather than a real heart signal
4. The 80% headroom (26000 instead of 32767) prevents clipping from any rounding artefacts


### Filter Ordering — Why This Sequence?

The order HP → Notch → LP → Normalize is not arbitrary:

1. **HP first** — removes DC offset and baseline drift before the notch and LP filters process the signal. A large DC component would cause the notch filter's internal state to saturate temporarily (transient response), corrupting the first ~50 samples of filtered output.

2. **Notch second** — the 50 Hz spike is narrow and strong. Removing it before the LP filter prevents any spectral leakage from that spike from being smeared into adjacent frequencies by the LP's transition band.

3. **LP third** — removes all remaining high-frequency content after the notch has cleaned up 50 Hz.

4. **Normalize last** — always runs on the fully cleaned signal, so the gain calculation reflects only genuine cardiac content, not noise that might have been present before filtering.

<hr>

## Real-Time Dashboard

The dashboard (`dashboard.html`) is a single HTML file served by Flask. It connects to the server via `fetch()` and updates all charts in real time.

**Features:**

| Section | What It Shows |
|---|---|
| **Stats Bar** | Total records, Normal count, Abnormal count, Avg confidence, RMS dB, Spectral Centroid |
| **Live Diagnosis** | Prediction label, confidence %, medical message, RMS/ZCR/duration metrics |
| **Audio Waveform** | Real 200-point downsampled waveform from the ESP32 recording |
| **Class Probability** | Horizontal bar chart — probability for each of 4 classes |
| **Confidence Trend** | Line chart — confidence over the last 25 predictions |
| **Session Distribution** | Donut chart — breakdown of prediction classes this session |
| **MFCC Heatmap** | Real 40×60 MFCC matrix rendered as a colour heatmap canvas |
| **Patient Form** | Name, Age, Gender, BP, HR, Weight, Height, Doctor, Notes |
| **Session History** | Table of all saved records with prediction + vitals |
| **PDF Export** | Full A4 clinical report with all charts, patient info, history table |

**Connection states:**
- `CONNECTED` — Flask server is running and receiving ESP32 data
- `DEMO MODE` — Server not running; dashboard self-simulates with realistic data
- `OFFLINE` — Was connected but server stopped

<hr>

## Dataset

**Recommended training datasets:**

| Dataset | Details |
|---|---|
| [PhysioNet 2016 Challenge](https://physionet.org/content/challenge-2016/) | 3,240 heart sound recordings, 5 classes — the standard benchmark |
| [PhysioNet CinC 2022](https://physionet.org/content/circor-heart-sound/1.0.3/) | Latest dataset with murmur grading |
| [ICBHI 2017 Respiratory Sounds](https://bhichallenge.med.auth.gr/) | 920 lung sound recordings from 126 patients — crackles, wheezes, normal |
| [PASCAL Heart Sound Challenge](http://www.peterjbentley.com/heartchallenge/) | 800 recordings — systolic, diastolic, normal, noisy |

**Folder structure for training:**
```
dataset/
├── normal/         ← .wav files of normal heart sounds
├── murmur/         ← .wav files with murmur
├── arrhythmia/     ← .wav files with irregular rhythm
└── extrasound/     ← .wav files with extra S3/S4 sounds
```

<hr>

## Results

Trained on PhysioNet 2016 heart sound dataset with Random Forest (200 trees):

| Metric | Score |
|---|---|
| Training accuracy | ~91% |
| Test accuracy | ~88% |
| Avg prediction latency | <200ms (local PC) |
| Audio capture time | 3 seconds |
| Total round-trip time | ~4–5 seconds |

End-to-end latency from ESP32 button press to prediction shown on dashboard: **~5 seconds**

<hr>

## Setup Guide

**1. Install Python dependencies:**
```bash
pip install flask flask-cors numpy librosa scikit-learn soundfile joblib tensorflow
```

**2. Train the model (once):**
```bash
python train_local.py
```

**3. Start the prediction server:**
```bash
python local_prediction_server.py
```

**4. Find your PC's local IP:**
```bash
# Windows
ipconfig
# Linux/Mac
ip addr
```

**5. Update `config.h` in PlatformIO project:**
```cpp
#define PC_IP "192.168.1.X"   // your PC's local IP
```

**6. Build & upload firmware via PlatformIO in VS Code:**
- Click ✔ (Build) then → (Upload) on the bottom blue bar

**7. Open dashboard:**
```
http://localhost:5000
```

**8. Test without ESP32:**
```
http://localhost:5000/test
```

> ⚠️ **Disclaimer:** This device is for **educational and research purposes only**. It is NOT a certified medical device and must NOT be used for clinical diagnosis. Always consult a qualified healthcare professional.

<hr>

<p align="center">
   ESP32 · Python · Flask · librosa 
</p>
<p align="center">
   Thank You !
</p>
