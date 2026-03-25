// =====================================================================
// CardioScope — Digital Stethoscope
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
// │                                                               │
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
// │  (no external resistor needed — uses internal pullup)        │
// └──────────────────────────────────────────────────────────────┘
//
// NO LEDs — all status shown on LCD1602
//
// GAIN SETTING (3.3V supply):
//   MAX9814 @ 3.3V → output centre = ~1.65V
//   GAIN_TWO = ±2.048V  ✅ correct for 3.3V
//   Midpoint raw value = (1.65 / 2.048) × 32767 ≈ 26400
//
// RECORDING TIME = 5 SECONDS:
//   At 800 SPS × 5s = 4000 samples = 8000 bytes sent
//   5 seconds captures 4–6 full heartbeat cycles @ 60–80 BPM
//   This is the clinical standard minimum for auscultation
//   Short enough for responsiveness, long enough for ML accuracy
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
#define PC_IP          "172.28.245.181"   // your PC local IP
#define SERVER_PORT    5000
// ===========================

// I2C pins
#define SDA_PIN  21
#define SCL_PIN  22

// Button only — no LEDs
#define BUTTON_PIN  15

// LCD1602 I2C
// Change to 0x3F if screen stays blank
#define LCD_ADDRESS  0x27
#define LCD_COLS     16
#define LCD_ROWS      2

// ADS1115 at 3.3V — GAIN_TWO is correct
#define ADS_GAIN      GAIN_TWO           // ±2.048V range
#define ADS_RATE      RATE_ADS1115_860SPS
// MAX9814 @ 3.3V: midpoint = (1.65/2.048)*32767 = 26400
#define ADC_MIDPOINT  26400

// === RECORDING TIME ===
// 5 seconds = best balance for heart sound classification
// Captures 4–6 full heartbeat cycles @ 60–80 BPM
// 800 SPS × 5s = 4000 samples = 8000 bytes over WiFi
#define SAMPLE_RATE     800
#define RECORD_SECONDS  5
#define TOTAL_SAMPLES   (SAMPLE_RATE * RECORD_SECONDS)  // 4000

// ===== OBJECTS =====
Adafruit_ADS1115  ads;
LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLS, LCD_ROWS);

// Audio buffer
int16_t audioBuffer[TOTAL_SAMPLES];

// Prediction state
String lastPrediction = "STANDBY";
String lastConfidence = "--";
String lastMessage    = "Press BTN to start";
bool   isAbnormal     = false;

// Button state for edge detection
bool btnPrev     = false;
bool btnTriggered = false;

// =====================================================================
// CUSTOM LCD CHARACTERS
// =====================================================================
byte heartChar[8] = {                     // ♥  slot 0
  0b00000, 0b01010, 0b11111,
  0b11111, 0b01110, 0b00100,
  0b00000, 0b00000
};
byte wifiChar[8] = {                      // WiFi arc  slot 1
  0b00000, 0b01110, 0b10001,
  0b00100, 0b01010, 0b00000,
  0b00100, 0b00000
};
byte warnChar[8] = {                      // ⚠  slot 2
  0b00100, 0b01110, 0b01110,
  0b11111, 0b11111, 0b00000,
  0b00100, 0b00000
};
byte recChar[8] = {                       // ● rec dot  slot 3
  0b00000, 0b01110, 0b11111,
  0b11111, 0b11111, 0b01110,
  0b00000, 0b00000
};
byte okChar[8] = {                        // ✓  slot 4
  0b00000, 0b00001, 0b00011,
  0b10110, 0b11100, 0b01000,
  0b00000, 0b00000
};

// =====================================================================
// LCD HELPERS
// =====================================================================

// Print text exactly filling 'width' columns — clears leftover chars
void lcdPrint(int col, int row, String text, int width = 16) {
  lcd.setCursor(col, row);
  if ((int)text.length() > width)
    text = text.substring(0, width);
  while ((int)text.length() < width)
    text += " ";
  lcd.print(text);
}

// Clear both rows
void lcdClear() {
  lcd.clear();
}

// ── Boot ──────────────────────────────────────────────────────────────
void lcdShowBoot(String l1, String l2 = "") {
  lcdClear();
  lcdPrint(0, 0, l1);
  lcdPrint(0, 1, l2);
}

// ── WiFi connecting ──────────────────────────────────────────────────
void lcdShowConnecting(int attempt) {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(1));  // wifi icon
  lcd.print(" Connecting...");
  String bar = "";
  for (int i = 0; i < (attempt % 4); i++) bar += ".";
  lcdPrint(0, 1, "Attempt " + String(attempt) + bar);
}

// ── WiFi connected ───────────────────────────────────────────────────
void lcdShowConnected(String ip) {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(1));  // wifi icon
  lcd.print(" WiFi OK!");
  lcdPrint(0, 1, ip);
}

// ── WiFi failed ──────────────────────────────────────────────────────
void lcdShowWiFiFailed() {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(2));  // warn
  lcd.print(" WiFi FAILED");
  lcdPrint(0, 1, "Check settings");
}

// ── Ready / standby ──────────────────────────────────────────────────
void lcdShowStandby() {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(0));  // heart
  lcd.print(" ");
  // Fit prediction in 13 chars
  String p = lastPrediction;
  if (p.length() > 13) p = p.substring(0, 13);
  lcd.print(p);
  lcdPrint(0, 1, "Conf:" + lastConfidence + " [BTN]");
}

// ── Recording ────────────────────────────────────────────────────────
void lcdShowRecording(int samplesDone, int total, int secondsDone) {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(3));  // rec dot
  lcd.print(" REC ");
  lcd.print(secondsDone);
  lcd.print("/");
  lcd.print(RECORD_SECONDS);
  lcd.print("s...");

  // Progress bar: 14 blocks
  int filled = map(samplesDone, 0, total, 0, 14);
  lcd.setCursor(0, 1);
  lcd.print("[");
  for (int i = 0; i < 14; i++)
    lcd.print(i < filled ? (char)255 : ' ');
  lcd.print("]");
}

// ── Sending to PC ────────────────────────────────────────────────────
void lcdShowSending() {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(1));  // wifi icon
  lcd.print(" Sending...");
  lcdPrint(0, 1, "ML analysing...");
}

// ── Prediction result ────────────────────────────────────────────────
void lcdShowResult(String pred, String conf, bool abnormal) {
  lcdClear();
  lcd.setCursor(0, 0);
  if (abnormal)
    lcd.write(byte(2));  // warn
  else
    lcd.write(byte(4));  // ok checkmark
  lcd.print(" ");
  String p = pred;
  if (p.length() > 13) p = p.substring(0, 13);
  lcd.print(p);
  lcdPrint(0, 1, "Conf: " + conf);
}

// ── Medical message ──────────────────────────────────────────────────
// Scrolls a long message across row 1
void lcdShowMessage(String msg) {
  // Row 0: prediction + confidence
  lcdClear();
  String row0 = lastPrediction;
  if (row0.length() > 9) row0 = row0.substring(0, 9);
  row0 += " " + lastConfidence;
  lcdPrint(0, 0, row0);

  // Row 1: message, scroll if > 16 chars
  if (msg.length() <= 16) {
    lcdPrint(0, 1, msg);
    delay(3000);
  } else {
    // Scroll message
    String padded = msg + "  ";
    for (int i = 0; i <= (int)padded.length() - 16; i++) {
      lcd.setCursor(0, 1);
      lcd.print(padded.substring(i, i + 16));
      delay(350);
    }
  }
}

// ── Error ────────────────────────────────────────────────────────────
void lcdShowError(String err) {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(2));  // warn
  lcd.print(" ERROR");
  lcdPrint(0, 1, err);
}

// ── HTTP error with code ─────────────────────────────────────────────
void lcdShowHTTPError(int code) {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(2));
  lcd.print(" Server Error");
  lcdPrint(0, 1, "HTTP code: " + String(code));
}

// ── No WiFi ──────────────────────────────────────────────────────────
void lcdShowNoWiFi() {
  lcdClear();
  lcd.setCursor(0, 0);
  lcd.write(byte(2));
  lcd.print(" WiFi Lost!");
  lcdPrint(0, 1, "Reconnecting...");
}

// =====================================================================
// ADS1115 INIT
// =====================================================================
void setupADS() {
  lcdShowBoot("ADS1115 init...", "GAIN_TWO / 3.3V");

  if (!ads.begin(0x48)) {
    Serial.println("[ERROR] ADS1115 not found at 0x48!");
    lcdShowError("ADS1115 missing");
    // Halt — cannot record without ADC
    while (true) delay(1000);
  }

  ads.setGain(ADS_GAIN);
  ads.setDataRate(ADS_RATE);

  Serial.println("[OK] ADS1115 ready");
  Serial.println("[OK] Gain: GAIN_TWO (+-2.048V) — 3.3V supply");
  Serial.printf ("[OK] Midpoint offset: %d\n", ADC_MIDPOINT);

  lcdShowBoot("ADS1115 OK!", "GAIN_TWO +-2.048V");
  delay(1000);
}

// =====================================================================
// WI-FI INIT
// =====================================================================
void setupWiFi() {
  lcdShowConnecting(0);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
    lcdShowConnecting(attempts);
  }

  if (WiFi.status() == WL_CONNECTED) {
    String ip = WiFi.localIP().toString();
    Serial.printf("\n[WiFi] Connected! IP: %s\n", ip.c_str());
    lcdShowConnected(ip);
    delay(2500);
  } else {
    Serial.println("\n[ERROR] WiFi connection failed!");
    lcdShowWiFiFailed();
    delay(2000);
    // Don't halt — button press will retry
  }
}

// =====================================================================
// RECORD AUDIO — 5 seconds at 800 SPS
// Button already pressed before this is called
// =====================================================================
bool recordAudio() {
  Serial.printf("[REC] Starting %ds @ %d SPS...\n",
                RECORD_SECONDS, SAMPLE_RATE);

  const unsigned long intervalUs = 1000000UL / SAMPLE_RATE; // 1250 µs
  unsigned long nextSample       = micros();
  int clipped = 0;
  int lastSecShown = -1;

  lcdShowRecording(0, TOTAL_SAMPLES, 0);

  for (int i = 0; i < TOTAL_SAMPLES; i++) {

    // Precise sample timing
    while (micros() < nextSample) { /* spin-wait */ }
    nextSample += intervalUs;

    // Read ADS1115 channel A0
    int16_t raw = ads.readADC_SingleEnded(0);

    // Clip detection
    if (raw >= 32760 || raw <= -32760) clipped++;

    // Remove DC midpoint → centre around zero
    audioBuffer[i] = (int16_t)constrain(
                       (int32_t)raw - ADC_MIDPOINT,
                       -32768, 32767);

    // Update LCD every second (every 800 samples)
    int secDone = i / SAMPLE_RATE;
    if (secDone != lastSecShown) {
      lastSecShown = secDone;
      lcdShowRecording(i, TOTAL_SAMPLES, secDone);
    }
  }

  // Show 100% done
  lcdShowRecording(TOTAL_SAMPLES, TOTAL_SAMPLES, RECORD_SECONDS);
  delay(300);

  Serial.printf("[REC] Done! %d samples | clipped: %d (%.1f%%)\n",
                TOTAL_SAMPLES, clipped,
                (float)clipped / TOTAL_SAMPLES * 100.0f);

  // Quality check
  if ((float)clipped / TOTAL_SAMPLES > 0.10f) {
    Serial.println("[WARN] >10% samples clipped!");
    Serial.println("       Move mic slightly away from chest piece.");
    lcdShowError("Signal too loud!");
    delay(2000);
    return false;  // bad recording — don't send
  }

  // Check if signal is just noise (all near zero = mic not working)
  long sumAbs = 0;
  for (int i = 0; i < TOTAL_SAMPLES; i++)
    sumAbs += abs(audioBuffer[i]);
  long avgAbs = sumAbs / TOTAL_SAMPLES;

  if (avgAbs < 20) {
    Serial.println("[WARN] Signal too quiet — check mic connection!");
    lcdShowError("No signal! ChkMic");
    delay(2000);
    return false;
  }

  Serial.printf("[REC] Avg signal level: %ld (good)\n", avgAbs);
  return true;  // recording OK
}

// =====================================================================
// SEND TO PC & GET PREDICTION
// =====================================================================
void sendAndPredict() {
  // Check WiFi first
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ERROR] WiFi not connected!");
    lcdShowNoWiFi();
    // Try to reconnect
    WiFi.reconnect();
    int wait = 0;
    while (WiFi.status() != WL_CONNECTED && wait < 10) {
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

  Serial.printf("[HTTP] POST → %s\n", url.c_str());
  Serial.printf("[HTTP] Payload: %d samples × 2 bytes = %d bytes\n",
                TOTAL_SAMPLES, TOTAL_SAMPLES * 2);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type",    "application/octet-stream");
  http.addHeader("X-Sample-Rate",   String(SAMPLE_RATE));    // 800
  http.addHeader("X-Record-Secs",   String(RECORD_SECONDS)); // 5
  http.addHeader("X-Mic-Type",      "MAX9814-ADS1115-3V3");
  http.addHeader("X-ADC-Gain",      "GAIN_TWO");
  http.setTimeout(25000);  // 25s timeout — enough for 5s audio + ML

  int code = http.POST((uint8_t*)audioBuffer,
                        TOTAL_SAMPLES * sizeof(int16_t));

  if (code == 200) {
    String resp = http.getString();
    Serial.println("[HTTP] 200 OK");
    Serial.println("[HTTP] Response: " + resp);

    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, resp);

    if (!err) {
      const char* pred = doc["prediction"] | "unknown";
      float       conf = doc["confidence"]  | 0.0f;
      const char* msg  = doc["message"]     | "";

      // Store result
      lastPrediction = String(pred);
      lastPrediction.toUpperCase();
      lastConfidence = String(conf * 100.0f, 1) + "%";
      lastMessage    = String(msg);
      isAbnormal     = (String(pred) != "normal");

      // Serial Monitor output
      Serial.println();
      Serial.println("╔══════════════════════════╗");
      Serial.printf ("║ RESULT    : %-13s ║\n", pred);
      Serial.printf ("║ CONFIDENCE: %-13s ║\n",
                     (String(conf*100.0f,1)+"%").c_str());
      Serial.printf ("║ ABNORMAL  : %-13s ║\n",
                     isAbnormal ? "YES !" : "No");
      Serial.println("╚══════════════════════════╝");
      Serial.println(msg);
      Serial.println();

      // Show result on LCD
      lcdShowResult(lastPrediction, lastConfidence, isAbnormal);
      delay(4000);

      // Show scrolling medical message
      lcdShowMessage(lastMessage);

    } else {
      Serial.println("[ERROR] JSON parse failed: " + resp);
      lcdShowError("Bad JSON resp");
    }

  } else if (code == -1) {
    Serial.println("[ERROR] Connection refused — is server running?");
    lcdShowError("Server offline?");
  } else {
    Serial.printf("[ERROR] HTTP %d\n", code);
    lcdShowHTTPError(code);
  }

  http.end();
}

// =====================================================================
// SETUP — runs once on power-on
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  // Button — internal pullup, reads LOW when pressed
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // I2C bus — 3.3V side
  // ADS1115 connects here directly
  // LCD connects through level shifter (LV side here)
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);  // 100kHz — safe through level shifter

  lcd.init();
  lcd.backlight();

  // Register custom characters into LCD CGRAM
  lcd.createChar(0, heartChar);  // ♥
  lcd.createChar(1, wifiChar);   // wifi
  lcd.createChar(2, warnChar);   // ⚠
  lcd.createChar(3, recChar);    // ● rec
  lcd.createChar(4, okChar);     // ✓

  // Splash screen
  lcdShowBoot("CardioScope v2.2", "Starting up...");
  delay(1200);

  // Print system info to Serial Monitor
  Serial.println();
  Serial.println("╔════════════════════════════════════╗");
  Serial.println("║   CardioScope — Digital Stethoscope ║");
  Serial.println("║   ESP32 + MAX9814 + ADS1115 + LCD   ║");
  Serial.println("╠════════════════════════════════════╣");
  Serial.printf ("║   Sample Rate  : %d SPS             ║\n", SAMPLE_RATE);
  Serial.printf ("║   Record Time  : %d seconds          ║\n", RECORD_SECONDS);
  Serial.printf ("║   Total Samples: %d                 ║\n", TOTAL_SAMPLES);
  Serial.printf ("║   Payload Size : %d bytes           ║\n", TOTAL_SAMPLES*2);
  Serial.printf ("║   ADC Gain     : GAIN_TWO ±2.048V   ║\n");
  Serial.printf ("║   Midpoint     : %d                ║\n", ADC_MIDPOINT);
  Serial.println("╚════════════════════════════════════╝");
  Serial.println();

  // Init ADS1115
  setupADS();

  // Connect WiFi
  setupWiFi();

  // Show ready screen
  lcdShowBoot("Press BUTTON to", "record 5s audio");
  delay(1500);
  lcdShowStandby();

  Serial.println("[READY] Waiting for button press...");
  Serial.println("[INFO]  Press button to start recording.");
}

// =====================================================================
// LOOP — runs forever, button edge detection
// =====================================================================
void loop() {
  // Read button — LOW = pressed (internal pullup)
  bool btnNow = (digitalRead(BUTTON_PIN) == LOW);

  // Edge detection: only trigger on the FALLING EDGE
  // (moment button goes from not-pressed to pressed)
  // This prevents repeated triggers while holding the button
  if (btnNow && !btnPrev) {
    btnTriggered = true;
  }
  btnPrev = btnNow;

  if (btnTriggered) {
    btnTriggered = false;  // reset immediately so it won't re-fire

    Serial.println("[BTN] Button pressed — starting cycle");

    // Show "Preparing..."
    lcdClear();
    lcd.setCursor(0, 0);
    lcd.write(byte(3));  // rec dot
    lcd.print(" Preparing...");
    lcdPrint(0, 1, "Place stethoscope");
    delay(1500);  // give user time to position stethoscope

    // Record audio — returns false if signal bad
    bool ok = recordAudio();

    if (ok) {
      // Send to Flask server and show prediction
      sendAndPredict();
      // Show result summary, then standby
      delay(3000);
    }

    // Back to standby screen
    lcdShowStandby();
    Serial.println("[READY] Waiting for next button press...");
  }

  // Small delay to prevent button bounce loop
  delay(10);
}
