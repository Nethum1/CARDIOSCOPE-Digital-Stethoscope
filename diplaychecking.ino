#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define MIC_PIN 34
#define THRESHOLD 1900

LiquidCrystal_I2C lcd(0x27,16,2);

unsigned long detectionTime = 0;
bool soundDetected = false;

void setup() {

  Serial.begin(115200);

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0,0);
  lcd.print("Heart Monitor");
  delay(2000);
  lcd.clear();

  lcd.setCursor(0,0);
  lcd.print("Listening...");
}

void loop() {

  int micValue = analogRead(MIC_PIN);
  Serial.println(micValue);

  if(micValue > THRESHOLD) {

    soundDetected = true;
    detectionTime = millis();

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Heart Sound");
    lcd.setCursor(0,1);
    lcd.print("Detected!");
  }

  if(soundDetected && millis() - detectionTime > 10000) {

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Listening...");
    soundDetected = false;
  }

  delay(50);
}