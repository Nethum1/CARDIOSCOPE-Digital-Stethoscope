// =====================================================================
// CardioScope — Digital Stethoscope  v3.0
// ESP32 (VIN=5V) + MAX9814 (3.3V) + ADS1115 (3.3V)
// + LCD1602 (5V, I2C backpack PCF8574) + Level Shifter
//
// POWER WIRING:
// ┌──────────────────────────────────────────────────────────────┐
// │  5V supply  ──→  ESP32 VIN          (onboard reg → 3.3V)    │
// │  5V supply  ──→  LCD1602 VCC        (pin 2)                  │
// │  5V supply  ──→  LCD1602 A+         (pin 15, backlight)      │
// │  5V supply  ──→  Level Shifter HV                            │
// │  3.3V (from ESP32 pin) ──→  ADS1115 VDD                     │
// │  3.3V (from ESP32 pin) ──→  MAX9814 VDD                     │
// │  3.3V (from ESP32 pin) ──→  Level Shifter LV                │
// │  GND ──→ ALL GND                                             │
// └──────────────────────────────────────────────────────────────┘
//
// I2C WIRING:
// ┌──────────────────────────────────────────────────────────────┐
// │  ESP32 GPIO21 (SDA, 3.3V) ──→ Level Shifter LV_A            │
// │  ESP32 GPIO22 (SCL, 3.3V) ──→ Level Shifter LV_B            │
// │  Level Shifter HV_A       ──→ LCD1602 SDA  (5V)             │
// │  Level Shifter HV_B       ──→ LCD1602 SCL  (5V)             │
// │  ADS1115 SDA  ──→ ESP32 GPIO21  (3.3V direct, no shifter)   │
// │  ADS1115 SCL  ──→ ESP32 GPIO22  (3.3V direct, no shifter)   │
// │  ADS1115 ADDR ──→ GND           (I2C address = 0x48)        │
// │  ADS1115 A0   ──→ MAX9814 OUT                                │
// └──────────────────────────────────────────────────────────────┘
//
// BUTTON:
// ┌──────────────────────────────────────────────────────────────┐
// │  Button one side ──→ ESP32 GPIO15                            │
// │  Button other    ──→ GND                                     │
// └──────────────────────────────────────────────────────────────┘
//
// DSP PIPELINE (applied after recording, before sending):
// ┌──────────────────────────────────────────────────────────────┐
// │  1. High-pass  @ 20 Hz  → removes DC drift + breathing       │
// │  2. Notch      @ 50 Hz  → removes power line interference    │
// │  3. Low-pass   @ 300 Hz → removes high-frequency noise       │
// │  4. Normalize           → scales to 80% full 16-bit range    │
// └──────────────────────────────────────────────────────────────┘
//
// LCD I2C address: 0x27 (try 0x3F if blank screen)
// ADS1115 I2C address: 0x48
//
// platformio.ini lib_deps:
//   bblanchon/ArduinoJson @ ^7.0.0
//   adafruit/Adafruit ADS1X15 @ ^2.5.0
//   johnrickman/LiquidCrystal I2C @ ^1.1.4
//   adafruit/Adafruit BusIO @ ^1.14.5
// =====================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <LiquidCrystal_I2C.h>

// ===== CONFIGURE THESE =====
#define WIFI_SSID      "Redmi 10A Sport"
#define WIFI_PASSWORD  "254860neth"
#define PC_IP          "172.28.245.181"
#define SERVER_PORT    5000
// ===========================

#define SDA_PIN     21
#define SCL_PIN     22
#define BUTTON_PIN  15

#define LCD_ADDRESS  0x27
#define LCD_COLS     16
#define LCD_ROWS      2

#define ADS_GAIN      GAIN_TWO          // ±2.048V — correct for 3.3V
#define ADS_RATE      RATE_ADS1115_860SPS
#define ADC_MIDPOINT  26400             // (1.65V / 2.048V) × 32767

#define SAMPLE_RATE     800
#define RECORD_SECONDS  5
#define TOTAL_SAMPLES   (SAMPLE_RATE * RECORD_SECONDS)  // 4000

// =====================================================================
// DSP — IIR BIQUAD FILTER COEFFICIENTS  (all @ Fs = 800 Hz)
// =====================================================================
//
// Transfer function per stage:
//   y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2]
//                  - a1·y[n-1] - a2·y[n-2]
//
// ── 1. High-pass Butterworth 2nd order @ 20 Hz ──────────────────
//    Removes DC offset, breathing artifacts, low-frequency rumble
static const float HP_B0 =  0.89442f;
static const float HP_B1 = -1.78885f;
static const float HP_B2 =  0.89442f;
static const float HP_A1 = -1.77822f;
static const float HP_A2 =  0.79990f;

// ── 2. Notch filter @ 50 Hz, Q = 35 ────────────────────────────
//    Removes mains power line interference
static const float NOTCH_B0 =  0.99456f;
static const float NOTCH_B1 = -1.83817f;
static const float NOTCH_B2 =  0.99456f;
static const float NOTCH_A1 = -1.83817f;
static const float NOTCH_A2 =  0.98912f;

// ── 3. Low-pass Butterworth 2nd order @ 300 Hz ──────────────────
//    Removes high-frequency noise — heart sounds live at 20–300 Hz
static const float LP_B0 =  0.56901f;
static const float LP_B1 =  1.13803f;
static const float LP_B2 =  0.56901f;
static const float LP_A1 =  0.94281f;
static const float LP_A2 =  0.33333f;

struct BiquadState {
  float x1 = 0, x2 = 0;
  float y1 = 0, y2 = 0;
};

// =====================================================================
// OBJECTS + STATE
// =====================================================================
Adafruit_ADS1115  ads;
LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLS, LCD_ROWS);

int16_t audioBuffer[TOTAL_SAMPLES];

String lastPrediction = "STANDBY";
String lastConfidence = "--";
String lastMessage    = "Press BTN to start";
bool   isAbnormal     = false;

bool btnPrev      = false;
bool btnTriggered = false;

// =====================================================================
// CUSTOM LCD CHARACTERS
// =====================================================================
byte heartChar[8] = {
  0b00000, 0b01010, 0b11111,
  0b11111, 0b01110, 0b00100,
  0b00000, 0b00000
};
byte wifiChar[8] = {
  0b00000, 0b01110, 0b10001,
  0b00100, 0b01010, 0b00000,
  0b00100, 0b00000
};
byte warnChar[8] = {
  0b00100, 0b01110, 0b01110,
  0b11111, 0b11111, 0b00000,
  0b00100, 0b00000
};
byte recChar[8] = {
  0b00000, 0b01110, 0b11111,
  0b11111, 0b11111, 0b01110,
  0b00000, 0b00000
};
byte okChar[8] = {
  0b00000, 0b00001, 0b00011,
  0b10110, 0b11100, 0b01000,
  0b00000, 0b00000
};

// =====================================================================
// LCD HELPERS
// =====================================================================
void lcdPrint(int col, int row, String text, int width = 16) {
  lcd.setCursor(col, row);
  if ((int)text.length() > width) text = text.substring(0, width);
  while ((int)text.length() < width) text += " ";
  lcd.print(text);
}
void lcdClear() { lcd.clear(); }

void lcdShowBoot(String l1, String l2 = "") {
  lcdClear(); lcdPrint(0, 0, l1); lcdPrint(0, 1, l2);
}
void lcdShowConnecting(int attempt) {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(1)); lcd.print(" Connecting...");
  String bar = "";
  for (int i = 0; i < (attempt % 4); i++) bar += ".";
  lcdPrint(0, 1, "Attempt " + String(attempt) + bar);
}
void lcdShowConnected(String ip) {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(1)); lcd.print(" WiFi OK!");
  lcdPrint(0, 1, ip);
}
void lcdShowWiFiFailed() {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(2)); lcd.print(" WiFi FAILED");
  lcdPrint(0, 1, "Check settings");
}
void lcdShowStandby() {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(0)); lcd.print(" ");
  String p = lastPrediction;
  if (p.length() > 13) p = p.substring(0, 13);
  lcd.print(p);
  lcdPrint(0, 1, "Conf:" + lastConfidence + " [BTN]");
}
void lcdShowRecording(int samplesDone, int total, int secondsDone) {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(3)); lcd.print(" REC ");
  lcd.print(secondsDone); lcd.print("/"); lcd.print(RECORD_SECONDS); lcd.print("s...");
  int filled = map(samplesDone, 0, total, 0, 14);
  lcd.setCursor(0, 1); lcd.print("[");
  for (int i = 0; i < 14; i++) lcd.print(i < filled ? (char)255 : ' ');
  lcd.print("]");
}
void lcdShowProcessing() {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(3)); lcd.print(" Processing...");
  lcdPrint(0, 1, "HP Notch LP Norm");
}
void lcdShowSending() {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(1)); lcd.print(" Sending...");
  lcdPrint(0, 1, "ML analysing...");
}
void lcdShowResult(String pred, String conf, bool abnormal) {
  lcdClear();
  lcd.setCursor(0, 0);
  if (abnormal) lcd.write(byte(2)); else lcd.write(byte(4));
  lcd.print(" ");
  String p = pred;
  if (p.length() > 13) p = p.substring(0, 13);
  lcd.print(p);
  lcdPrint(0, 1, "Conf: " + conf);
}
void lcdShowMessage(String msg) {
  lcdClear();
  String row0 = lastPrediction;
  if (row0.length() > 9) row0 = row0.substring(0, 9);
  row0 += " " + lastConfidence;
  lcdPrint(0, 0, row0);
  if (msg.length() <= 16) {
    lcdPrint(0, 1, msg); delay(3000);
  } else {
    String padded = msg + "  ";
    for (int i = 0; i <= (int)padded.length() - 16; i++) {
      lcd.setCursor(0, 1);
      lcd.print(padded.substring(i, i + 16));
      delay(350);
    }
  }
}
void lcdShowError(String err) {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(2)); lcd.print(" ERROR");
  lcdPrint(0, 1, err);
}
void lcdShowHTTPError(int code) {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(2)); lcd.print(" Server Error");
  lcdPrint(0, 1, "HTTP: " + String(code));
}
void lcdShowNoWiFi() {
  lcdClear();
  lcd.setCursor(0, 0); lcd.write(byte(2)); lcd.print(" WiFi Lost!");
  lcdPrint(0, 1, "Reconnecting...");
}

// =====================================================================
// MAGIC BOOT ANIMATION
// Shows CARDIOSCOPE 1.0v with typewriter + sparkle + heartbeat effects
// =====================================================================
void lcdShowMagicIntro() {

  // ── Phase 1: stars build outward ─────────────────────────────────
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("*");
  delay(120);
  lcd.setCursor(0, 0); lcd.print("*  *");
  delay(120);
  lcd.setCursor(0, 0); lcd.print("*    *");
  delay(120);
  lcd.setCursor(0, 0); lcd.print("*      *");
  delay(250);

  // ── Phase 2: typewriter effect for CARDIOSCOPE ───────────────────
  lcd.clear();
  String name = "CARDIOSCOPE";
  for (int i = 0; i <= name.length(); i++) {
    lcd.setCursor(2, 0);         // 2-char left padding → centres on 16 cols
    lcd.print(name.substring(0, i));
    if (i < name.length()) {
      lcd.write(byte(255));      // solid block █ as typing cursor
    }
    delay(i == 0 ? 80 : 105);
  }
  delay(300);

  // ── Phase 3: sparkle border animates around name ─────────────────
  const char* frames[] = {
    "* CARDIOSCOPE *",
    "+ CARDIOSCOPE +",
    "* CARDIOSCOPE *"
  };
  for (int f = 0; f < 3; f++) {
    lcd.setCursor(0, 0);
    lcd.print(frames[f]);
    delay(170);
  }

  // ── Phase 4: heart icons + name locked in, type version row 2 ────
  lcd.setCursor(0, 0);
  lcd.write(byte(0));            // ♥
  lcd.print(" CARDIOSCOPE ");
  lcd.write(byte(0));            // ♥

  delay(200);
  String ver = "     v 1.0";
  for (int i = 0; i <= ver.length(); i++) {
    lcd.setCursor(0, 1);
    lcd.print(ver.substring(0, i));
    if (i < ver.length()) lcd.write(byte(255));
    else {
      // erase trailing cursor when done
      lcd.print(" ");
    }
    delay(100);
  }
  delay(400);

  // ── Phase 5: heartbeat pulse — row 0 blinks 3 times ─────────────
  for (int p = 0; p < 3; p++) {
    lcd.setCursor(0, 0);
    lcd.print("                ");   // blank row
    delay(100);
    lcd.setCursor(0, 0);
    lcd.write(byte(0));              // ♥
    lcd.print(" CARDIOSCOPE ");
    lcd.write(byte(0));              // ♥
    delay(200);
  }
  delay(500);

  // ── Phase 6: scroll "Starting up..." into row 2 ──────────────────
  lcd.setCursor(0, 1);
  lcd.print("                ");     // clear version row
  String ready = "  Starting up...";
  for (int i = 0; i < (int)ready.length(); i++) {
    lcd.setCursor(i, 1);
    lcd.print(ready[i]);
    delay(55);
  }

  delay(1500);   // hold final screen
  lcd.clear();
}

// =====================================================================
// ADS1115 INIT
// =====================================================================
void setupADS() {
  lcdShowBoot("ADS1115 init...", "");
  if (!ads.begin(0x48)) {
    lcdShowError("ADS1115 missing");
    while (true) delay(1000);
  }
  ads.setGain(ADS_GAIN);
  ads.setDataRate(ADS_RATE);
  delay(1000);
}

// =====================================================================
// WI-FI INIT — fixed: disconnect + mode reset + auto-restart after 20s
// =====================================================================
void setupWiFi() {
  lcdShowConnecting(0);

  WiFi.disconnect(true);   // clear any stale connection state
  delay(1000);
  WiFi.mode(WIFI_STA);
  delay(1000);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    attempts++;
    lcdShowConnecting(attempts);

    if (attempts >= 40) {
      lcdShowError("Restarting...");
      delay(1500);
      ESP.restart();       // hard restart after 20s timeout
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    lcdShowConnected(WiFi.localIP().toString());
    delay(2500);
  } else {
    lcdShowWiFiFailed();
    delay(2000);
  }
}

// =====================================================================
// DSP — BIQUAD FILTER (processes audioBuffer in-place)
// =====================================================================
void applyBiquad(float b0, float b1, float b2,
                 float a1, float a2,
                 BiquadState &s) {
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    float x = (float)audioBuffer[i];
    float y = b0 * x
            + b1 * s.x1 + b2 * s.x2
            - a1 * s.y1 - a2 * s.y2;
    s.x2 = s.x1;  s.x1 = x;
    s.y2 = s.y1;  s.y1 = y;
    audioBuffer[i] = (int16_t)constrain((int32_t)y, -32768, 32767);
  }
}

// =====================================================================
// DSP — NORMALIZATION
// Scales peak to 80% of full 16-bit range (±26000).
// Gain capped at 20× so pure noise is never over-amplified.
// =====================================================================
void normalizeSignal() {
  int16_t peak = 0;
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    int16_t v = (int16_t)abs(audioBuffer[i]);
    if (v > peak) peak = v;
  }
  if (peak < 100) return;   // too quiet — skip normalize

  float gain = 26000.0f / (float)peak;
  if (gain > 20.0f) gain = 20.0f;

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    audioBuffer[i] = (int16_t)constrain(
                       (int32_t)((float)audioBuffer[i] * gain),
                       -32768, 32767);
  }
}

// =====================================================================
// DSP — FULL SIGNAL PROCESSING PIPELINE
//
// Runs AFTER recording, BEFORE sending to PC.
// Order: HP → Notch → LP → Normalize
//   HP first  → centres signal so notch & LP work correctly
//   Notch     → kills 50 Hz before LP can smear it
//   LP        → removes everything above 300 Hz
//   Normalize → stretch clean signal to full range for ML
// =====================================================================
void processSignal() {
  lcdShowProcessing();

  BiquadState s;

  // Step 1 — High-pass @ 20 Hz
  s = {0, 0, 0, 0};
  applyBiquad(HP_B0, HP_B1, HP_B2, HP_A1, HP_A2, s);

  // Step 2 — Notch @ 50 Hz
  s = {0, 0, 0, 0};
  applyBiquad(NOTCH_B0, NOTCH_B1, NOTCH_B2, NOTCH_A1, NOTCH_A2, s);

  // Step 3 — Low-pass @ 300 Hz
  s = {0, 0, 0, 0};
  applyBiquad(LP_B0, LP_B1, LP_B2, LP_A1, LP_A2, s);

  // Step 4 — Normalize amplitude to 80% of ±32767
  normalizeSignal();
}

// =====================================================================
// RECORD AUDIO — 5 seconds at 800 SPS
// =====================================================================
bool recordAudio() {
  const unsigned long intervalUs = 1000000UL / SAMPLE_RATE;
  unsigned long nextSample       = micros();
  int clipped    = 0;
  int lastSecShown = -1;

  lcdShowRecording(0, TOTAL_SAMPLES, 0);

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    while (micros() < nextSample) {}
    nextSample += intervalUs;

    int16_t raw = ads.readADC_SingleEnded(0);
    if (raw >= 32760 || raw <= -32760) clipped++;

    // Remove DC midpoint — centre around zero
    audioBuffer[i] = (int16_t)constrain(
                       (int32_t)raw - ADC_MIDPOINT,
                       -32768, 32767);

    int secDone = i / SAMPLE_RATE;
    if (secDone != lastSecShown) {
      lastSecShown = secDone;
      lcdShowRecording(i, TOTAL_SAMPLES, secDone);
    }
  }

  lcdShowRecording(TOTAL_SAMPLES, TOTAL_SAMPLES, RECORD_SECONDS);
  delay(300);

  // Quality check — too much clipping
  if ((float)clipped / TOTAL_SAMPLES > 0.10f) {
    lcdShowError("Too loud! Move mic");
    delay(2000);
    return false;
  }

  // Quality check — silence (mic not connected / no contact)
  long sumAbs = 0;
  for (int i = 0; i < TOTAL_SAMPLES; i++)
    sumAbs += abs(audioBuffer[i]);
  long avgAbs = sumAbs / TOTAL_SAMPLES;

  if (avgAbs < 20) {
    lcdShowError("No signal! ChkMic");
    delay(2000);
    return false;
  }

  return true;
}

// =====================================================================
// SEND TO PC & GET PREDICTION
// =====================================================================
void sendAndPredict() {
  if (WiFi.status() != WL_CONNECTED) {
    lcdShowNoWiFi();
    WiFi.reconnect();
    int wait = 0;
    while (WiFi.status() != WL_CONNECTED && wait < 20) {
      delay(500); wait++;
    }
    if (WiFi.status() != WL_CONNECTED) {
      lcdShowError("WiFi lost! Retry");
      delay(2000);
      return;
    }
  }

  lcdShowSending();

  String url = String("http://") + PC_IP + ":"
             + String(SERVER_PORT) + "/predict";

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type",   "application/octet-stream");
  http.addHeader("X-Sample-Rate",  String(SAMPLE_RATE));
  http.addHeader("X-Record-Secs",  String(RECORD_SECONDS));
  http.addHeader("X-Mic-Type",     "MAX9814-ADS1115-3V3");
  http.addHeader("X-ADC-Gain",     "GAIN_TWO");
  http.addHeader("X-DSP-Pipeline", "HP20-NOTCH50-LP300-NORM");
  http.setTimeout(25000);

  int code = http.POST((uint8_t*)audioBuffer,
                        TOTAL_SAMPLES * sizeof(int16_t));

  if (code == 200) {
    String resp = http.getString();
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, resp);

    if (!err) {
      const char* pred = doc["prediction"] | "unknown";
      float       conf = doc["confidence"]  | 0.0f;
      const char* msg  = doc["message"]     | "";

      lastPrediction = String(pred);
      lastPrediction.toUpperCase();
      lastConfidence = String(conf * 100.0f, 1) + "%";
      lastMessage    = String(msg);
      isAbnormal     = (String(pred) != "normal");

      lcdShowResult(lastPrediction, lastConfidence, isAbnormal);
      delay(4000);
      lcdShowMessage(lastMessage);

    } else {
      lcdShowError("Bad JSON resp");
    }

  } else if (code == -1) {
    lcdShowError("Server offline?");
  } else {
    lcdShowHTTPError(code);
  }

  http.end();
}

// =====================================================================
// SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  lcd.init();
  lcd.backlight();

  // Custom characters must be registered BEFORE lcdShowMagicIntro()
  // because the animation uses the heart character (slot 0)
  lcd.createChar(0, heartChar);  // ♥
  lcd.createChar(1, wifiChar);   // wifi
  lcd.createChar(2, warnChar);   // ⚠
  lcd.createChar(3, recChar);    // ● rec
  lcd.createChar(4, okChar);     // ✓

  // ── Magic boot animation ──────────────────────────────────────
  lcdShowMagicIntro();

  // ── Hardware init ─────────────────────────────────────────────
  setupADS();

  // ── WiFi init ─────────────────────────────────────────────────
  setupWiFi();

  // ── Ready screen ──────────────────────────────────────────────
  lcdShowBoot("Press BUTTON to", "record 5s audio");
  delay(1500);
  lcdShowStandby();

  Serial.println("[READY] CardioScope v3.0 — DSP enabled");

  // ── FIX: snapshot real pin state so the first loop() tick      ──
  // ── never sees a false rising edge and auto-triggers recording. ──
  btnPrev = (digitalRead(BUTTON_PIN) == LOW);
}

// =====================================================================
// LOOP
// =====================================================================
void loop() {
  bool btnNow = (digitalRead(BUTTON_PIN) == LOW);

  if (btnNow && !btnPrev) btnTriggered = true;
  btnPrev = btnNow;

  if (btnTriggered) {
    btnTriggered = false;

    lcdClear();
    lcd.setCursor(0, 0); lcd.write(byte(3)); lcd.print(" Preparing...");
    lcdPrint(0, 1, "Place stethoscope");
    delay(2000);

    // Step 1 — Record raw audio
    bool ok = recordAudio();

    if (ok) {
      // Step 2 — Apply DSP: HP → Notch → LP → Normalize
      processSignal();

      // Step 3 — Send clean signal to ML server
      sendAndPredict();
      delay(3000);
    }

    lcdShowStandby();
    Serial.println("[READY] Waiting for next button press...");
  }

  delay(10);
}
