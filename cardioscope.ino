// ============================================
// Digital Stethoscope - Local PC ML Version
// ESP32 + INMP441 I2S Mic -> HTTP POST -> PC
// ============================================
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>

// ===== CONFIGURE THESE =====
#define WIFI_SSID      "YOUR_WIFI_NAME"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"
#define PC_IP          "192.168.1.5"   // <-- Your PC local IP here
#define SERVER_PORT    5000
// ===========================

// I2S Pins (INMP441)
#define I2S_WS   25
#define I2S_SCK  26
#define I2S_SD   22
#define I2S_PORT I2S_NUM_0

// LEDs & Button
#define LED_GREEN  2
#define LED_RED    4
#define BUTTON_PIN 15

// Audio Settings
#define SAMPLE_RATE     16000
#define RECORD_SECONDS  3
#define BUFFER_SIZE     512
#define TOTAL_SAMPLES   (SAMPLE_RATE * RECORD_SECONDS)  // 48000 samples

int16_t audioBuffer[TOTAL_SAMPLES];

// ===== SETUP I2S =====
void setupI2S() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false
  };
  i2s_pin_config_t pins = {
    .bck_io_num  = I2S_SCK,
    .ws_io_num   = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  Serial.println("I2S initialized");
}

// ===== RECORD AUDIO =====
void recordAudio() {
  Serial.println("Recording 3 seconds...");
  digitalWrite(LED_RED, HIGH);
  int total = 0;
  size_t bytesRead;
  int16_t temp[BUFFER_SIZE];
  while (total < TOTAL_SAMPLES) {
    i2s_read(I2S_PORT, temp, sizeof(temp), &bytesRead, portMAX_DELAY);
    int count = min((int)(bytesRead / 2), TOTAL_SAMPLES - total);
    memcpy(audioBuffer + total, temp, count * 2);
    total += count;
  }
  digitalWrite(LED_RED, LOW);
  Serial.println("Recording done!");
}

// ===== SEND TO LOCAL PC & GET PREDICTION =====
void sendAndPredict() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected!");
    return;
  }

  String url = "http://" + String(PC_IP) + ":" + String(SERVER_PORT) + "/predict";
  Serial.println("Sending to: " + url);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/octet-stream");
  http.setTimeout(15000);  // 15 second timeout

  // POST raw audio bytes
  int responseCode = http.POST(
    (uint8_t*)audioBuffer,
    TOTAL_SAMPLES * sizeof(int16_t)
  );

  if (responseCode == 200) {
    String response = http.getString();
    Serial.println("=== PREDICTION RESULT ===");
    Serial.println(response);

    // Parse JSON response
    StaticJsonDocument<512> doc;
    deserializeJson(doc, response);
    const char* prediction = doc["prediction"];
    float confidence = doc["confidence"];
    const char* message = doc["message"];

    Serial.printf("Result: %s (%.1f%%)\n", prediction, confidence * 100);
    Serial.println(message);
    Serial.println("========================");
  } else {
    Serial.printf("HTTP Error: %d\n", responseCode);
    Serial.println(http.getString());
  }

  http.end();
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Connect Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
  digitalWrite(LED_GREEN, HIGH);

  setupI2S();
  Serial.println("Ready! Press button OR auto-records every 10 seconds.");
}

// ===== LOOP =====
unsigned long lastTime = 0;
void loop() {
  bool btn = (digitalRead(BUTTON_PIN) == LOW);
  bool auto_trigger = (millis() - lastTime > 10000);

  if (btn || auto_trigger) {
    lastTime = millis();
    recordAudio();
    sendAndPredict();
    delay(1000);
  }
}