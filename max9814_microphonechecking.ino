#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // I2C address, 16 columns, 2 rows

void setup() {
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0,0);
  lcd.print("ESP32 LCD Test");

  lcd.setCursor(0,1);
  lcd.print("Working!");
}

void loop() {
}