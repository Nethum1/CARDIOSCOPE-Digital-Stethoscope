#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_ADS1X15.h>

// Initialize LCD (Address 0x27 is common)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Initialize ADS1115
Adafruit_ADS1115 ads;

// Adjust this threshold based on Serial Monitor readings
// ADS1115 range is 0 to 32767
const int THRESHOLD = 18000; 

unsigned long detectionTime = 0;
bool soundDetected = false;

void setup() {
  Serial.begin(115200);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.print("Digital Steth");
  
  // Initialize ADS1115
  // ads.begin() usually defaults to 0x48
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS1115!");
    lcd.setCursor(0,1);
    lcd.print("ADS Error!");
    while (1);
  }

  // Set gain for highest precision (1 bit = 0.1875mV)
  ads.setGain(GAIN_ONE); 

  delay(2000);
  lcd.clear();
  lcd.print("Listening...");
}

void loop() {
  // Read from Channel 0 (A0)
  int16_t micValue = ads.readADC_SingleEnded(0);
  
  // Print for debugging / Serial Plotter
  Serial.println(micValue);

  // Logic for sound detection
  if (micValue > THRESHOLD) {
    soundDetected = true;
    detectionTime = millis();

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Heart Sound");
    lcd.setCursor(0, 1);
    lcd.print("Detected!");
  }

  // Return to "Listening" after 2 seconds (shorter for responsiveness)
  if (soundDetected && (millis() - detectionTime > 2000)) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Listening...");
    soundDetected = false;
  }

  // Small delay to prevent I2C bus congestion
  // For actual audio analysis, you would remove this, 
  // but for simple detection, 10ms is fine.
  delay(10); 
}