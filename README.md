# 🩺CARDIOSCOPE-(Digital-Stethoscope)

## Introduction
This project builds a fully functional **Digital Stethoscope** system using an ESP32 microcontroller and INMP441 I2S digital microphone. The system captures heart and lung sounds, streams the raw audio wirelessly over Wi-Fi to a local PC, runs a Machine Learning model to classify the sounds, and delivers **real-time diagnostic predictions** through a live web dashboard — all without any cloud dependency.

The key advantages of this digital approach over a traditional mechanical stethoscope are:
- Real-time audio capture using a high-quality **I2S digital microphone** (INMP441)
- Wireless transmission to a local PC over the **same Wi-Fi network**
- **Machine Learning classification** — Normal, Murmur, Arrhythmia, Extra Sound
- Live web dashboard with **real waveform**, **MFCC heatmap**, probability charts
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
- Make a microphone - amplification module that can detect frequencies from 50Hz to 20kHz 
- Transmit the sound clip to an associated laptop wirelessly 
- Be able to detect the disease by using an associated Deep Learning model
<hr>

<h2>ESP32</h2>
<img src="https://cdn.shopify.com/s/files/1/0609/6011/2892/files/doc-esp32-pinout-reference-wroom-devkit.png?width=1384" alt="ESP32 Pinout"
  width="700"
  height="500"
  align="middle"

<p>The microcontroller that we are using in our project is the ESP32.
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
┌─────────────────┐     I2S Digital Audio     ┌──────────────────────┐
│  INMP441 Mic    │ ─────────────────────────> │   ESP32 DevKit       │
│  (16kHz, 16bit) │                            │   (PlatformIO)       │
└─────────────────┘                            └──────────┬───────────┘
                                                          │
                                              HTTP POST /predict
                                              (raw int16 bytes)
                                              Same Wi-Fi Network
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LOCAL PC — Flask Server (Python)                   │
│                                                                       │
│  Raw bytes → float32 audio → librosa feature extraction               │
│       ↓                                                               │
│  • MFCC (40 coeff × 60 frames)   — for heatmap + model input         │
│  • Waveform (200 downsampled pts) — for waveform chart               │
│  • RMS dB, ZCR, Spectral Centroid — for signal metrics               │
│       ↓                                                               │
│  Random Forest Classifier → prediction + all class probabilities      │
│       ↓                                                               │
│  /latest endpoint stores full result                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                   Dashboard polls /latest every 2s
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (dashboard.html)                     │
│                                                                       │
│  ● Real waveform chart          ● MFCC heatmap (40×60)              │
│  ● Class probability bars       ● Confidence trend line             │
│  ● Session distribution donut   ● Signal metrics (RMS, SC, ZCR)    │
│  ● Patient record form          ● PDF report export                 │
└──────────────────────────────────────────────────────────────────────┘
```

<hr>

## Hardware

### Components

| Component | Specification |
|---|---|
| **Microcontroller** | ESP32-WROOM-32 / ESP32-DevKitC v4 |
| **Microphone** | INMP441 I2S Digital Microphone (16kHz, 16-bit) |
| **Stethoscope** | Acoustic stethoscope — mic inserted into chest piece |
| **LEDs** | Green (Wi-Fi connected), Red (Recording active) |
| **Push Button** | Momentary button — manual recording trigger |
| **Power** | USB 5V or 3.7V Li-Po battery + TP4056 charger |
| **Resistors** | 220Ω × 2 (LED current limiting) |
| **Breadboard** | 830-point solderless breadboard |
| **Jumper Wires** | Male-to-male and male-to-female, 20cm |

<hr>


### Heart & Lung Sounds

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

The INMP441 captures sounds in the 20Hz–16kHz range at 16kHz sample rate. Heart sounds (20–600Hz) and lung sounds (up to 2000Hz) both fall well within the captured band. The ESP32 DMA buffers 3 seconds of audio (48,000 samples) before transmitting.

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
