// =====================================================
// Digital Stethoscope
// ESP32 (3.3V) + MAX9814 (3.3V) + ADS1115 (3.3V)
// + LCD1602 I2C (5V) + Level Shifter
// Power: 7.4V LiPo → Buck#1 5V (LCD) → Buck#2 3.3V (sensors)
//
// WIRING:
// ┌─────────────────────────────────────────────────────┐
// │  7.4V LiPo (+) ──→ ESP32 VIN                        │
// │  7.4V LiPo (+) ──→ Buck#1 IN+ → 5V OUT             │
// │  7.4V LiPo (+) ──→ Buck#2 IN+ → 3.3V OUT           │
// │  GND ──────────→ everything GND                     │
// │                                                      │
// │  5V rail  ──→ LCD1602 VCC (pin 2)                   │
// │  5V rail  ──→ LCD1602 A   (pin 15, backlight +)     │
// │  5V rail  ──→ Level Shifter HV                      │
// │                                                      │
// │  3.3V rail ──→ ADS1115 VDD                          │
// │  3.3V rail ──→ MAX9814 VDD                          │
// │  3.3V rail ──→ Level Shifter LV                     │
// │                                                      │
// │  Level Shifter                                       │
// │    LV_A ←→ ESP32 GPIO21    HV_A ←→ LCD1602 SDA     │
// │    LV_B ←→ ESP32 GPIO22    HV_B ←→ LCD1602 SCL     │
// │                                                      │
// │  ADS1115 (3.3V, direct to ESP32 — no shifter)       │
// │    SDA → ESP32 GPIO21      SCL → ESP32 GPIO22       │
// │    ADDR → GND (0x48)       A0  → MAX9814 OUT        │
// │                                                      │
// │  LCD1602 I2C backpack PCF8574                        │
// │    SDA → Level Shifter HV_A                         │
// │    SCL → Level Shifter HV_B                         │
// │    (address 0x27 — if not working try 0x3F)         │
// │                                                      │
// │  LED Green → GPIO 2   (+ 220Ω to GND)               │
// │  LED Red   → GPIO 4   (+ 220Ω to GND)               │
// │  Button    → GPIO 15  (other end to GND)            │
// └─────────────────────────────────────────────────────┘
//
// NOTE: ADS1115 at 3.3V → use GAIN_TWO (±2.048V)
//       MAX9814 at 3.3V → output centres at ~1.65V
//       Midpoint offset = (1.65/2.048)*32767 ≈ 26400
//
// LCD1602 display screens:
//   Boot      → "CardioScope"
//   WiFi      → connecting animation
//   Connected → IP address + server
//   Recording → progress + sample count
//   Sending   → uploading status
//   Result    → prediction + confidence
//   Standby   → last result summary
//   Error     → error message
// =====================================================

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <LiquidCrystal_I2C.h>

// ===== CONFIGURE THESE =====
#define WIFI_SSID      "YOUR_WIFI_NAME"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"
#define PC_IP          "192.168.1.5"
#define SERVER_PORT    5000
// ===========================

// I2C pins (3.3V side → through level shifter for LCD)
#define SDA_PIN   21
#define SCL_PIN   22

// LEDs & Button
#define LED_GREEN  2
#define LED_RED    4
#define BUTTON_PIN 15

// LCD1602 I2C
// Try 0x27 first. If LCD blank, change to 0x3F
#define LCD_ADDRESS  0x27
#define LCD_COLS     16
#define LCD_ROWS      2

// ADS1115 — 3.3V supply → GAIN_TWO
#define ADS_GAIN      GAIN_TWO          // ±2.048V range
#define ADS_RATE      RATE_ADS1115_860SPS
// MAX9814 @ 3.3V centres at ~1.65V
// (1.65 / 2.048) * 32767 ≈ 26400
#define ADC_MIDPOINT  26400

// Audio
#define SAMPLE_RATE     800
#define RECORD_SECONDS  3
#define TOTAL_SAMPLES   (SAMPLE_RATE * RECORD_SECONDS)   // 2400

// ===== OBJECTS =====
Adafruit_ADS1115   ads;
LiquidCrystal_I2C  lcd(LCD_ADDRESS, LCD_COLS, LCD_ROWS);

// Audio buffer
int16_t audioBuffer[TOTAL_SAMPLES];

// Last prediction
String lastPrediction = "STANDBY";
String lastConfidence = "--";
String lastMessage    = "No data yet";
bool   isAbnormal     = false;

// =====================================================
// CUSTOM LCD CHARACTERS
// =====================================================
// Heart symbol ♥
byte heartChar[8] = {
  0b00000, 0b01010, 0b11111,
  0b11111, 0b01110, 0b00100, 0b00000, 0b00000
};
// Signal bar
byte signalChar[8] = {
  0b00000, 0b00001, 0b00011,
  0b00111, 0b01111, 0b11111, 0b00000, 0b00000
};
// Warning !
byte warnChar[8] = {
  0b00100, 0b00100, 0b01110,
  0b01110, 0b11111, 0b11111, 0b00000, 0b00100
};

// =====================================================
// LCD HELPERS
// =====================================================

// Print a string padded/truncated to exactly 'width' chars
void lcdPrint(int col, int row, String text, int width = 16) {
  lcd.setCursor(col, row);
  // Truncate if too long
  if ((int)text.length() > width) text = text.substring(0, width);
  // Pad with spaces to clear old chars
  while ((int)text.length() < width) text += " ";
  lcd.print(text);
}

void lcdBoot(String line1, String line2 = "") {
  lcd.clear();
  lcdPrint(0, 0, line1);
  lcdPrint(0, 1, line2);
}

void lcdConnecting(int dots) {
  lcd.clear();
  lcdPrint(0, 0, "Connecting WiFi");
  String d = "";
  for (int i = 0; i < (dots % 4); i++) d += ".";
  lcdPrint(0, 1, d);
}

void lcdConnected(String ip) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(byte(1));  // signal char
  lcd.print(" WiFi OK!");
  lcdPrint(0, 1, ip);
}

void lcdRecording(int sample, int total) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(byte(0));  // heart char
  lcd.print(" RECORDING...");

  // Progress bar — 14 chars wide
  int progress = map(sample, 0, total, 0, 14);
  lcd.setCursor(0, 1);
  lcd.print("[");
  for (int i = 0; i < 14; i++) {
    lcd.print(i < progress ? (char)255 : ' ');
  }
  lcd.print("]");
}

void lcdSending() {
  lcd.clear();
  lcdPrint(0, 0, "Sending to PC...");
  lcdPrint(0, 1, "ML analysing...");
}

void lcdResult(String pred, String conf, bool abnormal) {
  lcd.clear();

  // Row 0: icon + prediction (max 14 chars)
  lcd.setCursor(0, 0);
  if (abnormal) {
    lcd.write(byte(2));   // warning char
    lcd.print(" ");
  } else {
    lcd.write(byte(0));   // heart char
    lcd.print(" ");
  }
  // Truncate prediction to 13 chars
  String p = pred;
  if (p.length() > 13) p = p.substring(0, 13);
  lcd.print(p);

  // Row 1: confidence
  lcdPrint(0, 1, "Conf: " + conf);
}

void lcdMessage(String msg) {
  // Show scrolling message on row 1 if too long
  lcd.clear();
  lcdPrint(0, 0, lastPrediction + " " + lastConfidence);
  // Truncate message to 16 chars for row 1
  String m = msg;
  if (m.length() > 16) m = m.substring(0, 16);
  lcdPrint(0, 1, m);
}

void lcdStandby() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(byte(0));  // heart
  lcd.print(" ");
  String p = lastPrediction;
  if (p.length() > 12) p = p.substring(0, 12);
  lcd.print(p);
  lcdPrint(0, 1, "Conf:" + lastConfidence + " BTN");
}

void lcdError(String err) {
  lcd.clear();
  lcdPrint(0, 0, "!! ERROR !!");
  lcdPrint(0, 1, err);
}

// =====================================================
// ADS1115 SETUP
// =====================================================
void setupADS() {
  if (!ads.begin(0x48)) {
    Serial.println("[ERROR] ADS1115 not found!");
    lcdError("ADS1115 missing");
    while (true) delay(1000);
  }
  ads.setGain(ADS_GAIN);
  ads.setDataRate(ADS_RATE);
  Serial.println("[OK] ADS1115 — GAIN_TWO (±2.048V) — 3.3V mode");
  Serial.printf("[OK] Midpoint offset: %d\n", ADC_MIDPOINT);
}

// =====================================================
// WI-FI SETUP
// =====================================================
void setupWiFi() {
  lcdConnecting(0);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
    lcdConnecting(attempts);
  }

  if (WiFi.status() == WL_CONNECTED) {
    String ip = WiFi.localIP().toString();
    Serial.printf("\n[WiFi] Connected! IP: %s\n", ip.c_str());
    digitalWrite(LED_GREEN, HIGH);
    lcdConnected(ip);
    delay(2500);
  } else {
    Serial.println("\n[ERROR] WiFi failed!");
    lcdError("WiFi FAILED");
    delay(2000);
  }
}

// =====================================================
// RECORD AUDIO
// =====================================================
void recordAudio() {
  Serial.printf("[REC] %ds @ %d SPS\n",
                RECORD_SECONDS, SAMPLE_RATE);
  digitalWrite(LED_RED, HIGH);

  const unsigned long interval = 1000000UL / SAMPLE_RATE;
  unsigned long nextSample = micros();
  int clipped = 0;

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    while (micros() < nextSample) {}
    nextSample += interval;

    int16_t raw = ads.readADC_SingleEnded(0);

    if (raw >= 32760 || raw <= -32760) clipped++;

    // Remove DC midpoint
    audioBuffer[i] = (int16_t)constrain(
                       (int32_t)raw - ADC_MIDPOINT,
                       -32768, 32767);

    // Update LCD progress every 120 samples (~15%)
    if (i % 120 == 0) lcdRecording(i, TOTAL_SAMPLES);
  }

  // Show 100% on finish
  lcdRecording(TOTAL_SAMPLES, TOTAL_SAMPLES);
  delay(200);

  digitalWrite(LED_RED, LOW);
  Serial.printf("[REC] Done! %d samples | clipped: %d (%.1f%%)\n",
                TOTAL_SAMPLES, clipped,
                (float)clipped / TOTAL_SAMPLES * 100.0f);

  if ((float)clipped / TOTAL_SAMPLES > 0.05f) {
    Serial.println("[WARN] >5% clipped — mic too close or gain too high");
  }
}

// =====================================================
// SEND & PREDICT
// =====================================================
void sendAndPredict() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ERROR] WiFi disconnected!");
    lcdError("WiFi lost!");
    return;
  }

  lcdSending();

  String url = String("http://") + PC_IP + ":"
             + String(SERVER_PORT) + "/predict";

  Serial.printf("[HTTP] POST → %s (%d bytes)\n",
                url.c_str(), TOTAL_SAMPLES * 2);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type",  "application/octet-stream");
  http.addHeader("X-Sample-Rate", String(SAMPLE_RATE));
  http.addHeader("X-Mic-Type",    "MAX9814-ADS1115-3V3");
  http.addHeader("X-ADC-Gain",    "GAIN_TWO");
  http.setTimeout(20000);

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

      Serial.println("\n==============================");
      Serial.printf("  RESULT    : %s\n",     pred);
      Serial.printf("  CONFIDENCE: %.1f%%\n", conf * 100.0f);
      Serial.printf("  MESSAGE   : %s\n",     msg);
      Serial.println("==============================\n");

      // Show result on LCD
      lcdResult(lastPrediction, lastConfidence, isAbnormal);
      delay(3000);

      // Then show medical message
      lcdMessage(lastMessage);

    } else {
      Serial.println("[ERROR] JSON parse failed");
      lcdError("JSON parse err");
    }

  } else {
    Serial.printf("[ERROR] HTTP %d\n", code);
    lcdError("HTTP " + String(code));
  }

  http.end();
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_RED,    OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED,   LOW);

  // I2C — 3.3V side (goes through level shifter to 5V LCD)
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);    // 100kHz safe with level shifter

  // LCD init
  lcd.init();
  lcd.backlight();
  lcd.createChar(0, heartChar);
  lcd.createChar(1, signalChar);
  lcd.createChar(2, warnChar);

  lcdBoot("CardioScope v2.1", "Initialising...");
  delay(1000);

  Serial.println("====================================");
  Serial.println("  Stethoscope — LCD1602 Edition");
  Serial.println("  MAX9814(3.3V) + ADS1115(3.3V)");
  Serial.println("  LCD1602(5V)   + Level Shifter");
  Serial.println("  Power: 7.4V LiPo + 2x Buck");
  Serial.println("====================================");

  lcdBoot("ADS1115 init...", "GAIN_TWO / 3.3V");
  setupADS();
  delay(800);

  setupWiFi();

  lcdStandby();
  Serial.println("[READY] Press button or wait 10s");
}

// =====================================================
// LOOP
// =====================================================
unsigned long lastTime = 0;

void loop() {
  bool btn   = (digitalRead(BUTTON_PIN) == LOW);
  bool auto_ = (millis() - lastTime > 10000);

  if (btn || auto_) {
    if (btn) delay(50);   // debounce
    lastTime = millis();

    recordAudio();
    sendAndPredict();

    // Hold result 5 seconds then back to standby
    delay(5000);
    lcdStandby();
    delay(500);
  }
}
