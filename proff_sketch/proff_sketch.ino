#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_ADS1X15.h>

/* * BNO055 QUATERNION & FINGER SENSOR CODE (ESP32-C3)
 * F0 uses ESP32-C3 ADC. F1-F4 use ADS1115 via I2C.
 * Features an I2C scanner to prevent initialization errors.
 */

// --- I2C CONFIG (Custom for C3) ---
#define I2C_SDA 4
#define I2C_SCL 5

// --- BNO055 CONFIG ---
#define BNO_RST 6  // Change to an available GPIO on your C3 setup
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);
bool bnoFound = false; // Flag to track BNO055 status

// --- ADS1115 CONFIG ---
Adafruit_ADS1115 ads; 
bool adsFound = false; // Flag to track ADS1115 status

// --- FINGER SENSOR CONFIG ---
#define NUM_FINGERS 5
#define F0_PIN 0 // ESP32-C3 ADC1 pin for F0 (Valid: 0, 1, 2, 3)

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // 1. Reset BNO055 Hardware
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW); delay(20);
  digitalWrite(BNO_RST, HIGH); delay(100);

  // 2. Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000); 
  Wire.setTimeOut(100);

  // 3. I2C Scanner Block
  Serial.println("Scanning I2C bus...");
  byte error, address;
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("Found device at 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      
      // Check for specific sensor addresses
      if (address == 0x29) bnoFound = true; 
      if (address == 0x48) adsFound = true;
    }
  }

  // 4. Conditional Initialization
  if (bnoFound) {
    Serial.println("Initializing BNO055...");
    bno.begin();
    bno.setExtCrystalUse(true);
  } else {
    Serial.println("BNO055 missing at 0x29. Skipping init.");
  }

  if (adsFound) {
    Serial.println("Initializing ADS1115...");
    ads.begin();
  } else {
    Serial.println("ADS1115 missing at 0x48. Skipping init.");
  }
  
  Serial.println("Setup complete. Starting loop...");
  delay(1000);
}

void loop() {
  // --- READ QUATERNIONS ---


// --- READ FINGERS ---
  for (int i = 0; i < NUM_FINGERS; i++) {
    int raw = 0;
    
    if (i == 0) {
      // F0 is on ESP32-C3 ADC, always read it
      raw = analogRead(F0_PIN);
    } else {
      // F1-F4 are on ADS1115 (Channels inverted: A3 to A0)
      if (adsFound) {
        raw = ads.readADC_SingleEnded(4 - i) >> 3; 
      } else {
        // Output zero if ADS1115 is disconnected
        raw = 0;
      }
    }

    // Append Finger Data
    Serial.print(" F"); Serial.print(i); Serial.print(":"); Serial.print(raw);
  }
  if (bnoFound) {
    imu::Quaternion quat = bno.getQuat();
    Serial.print("QW:");  Serial.print(quat.w(), 4);
    Serial.print(" QX:"); Serial.print(quat.x(), 4);
    Serial.print(" QY:"); Serial.print(quat.y(), 4);
    Serial.print(" QZ:"); Serial.print(quat.z(), 4);
  } else {
    // Output zeros if sensor is disconnected
    Serial.print("QW:0.0000 QX:0.0000 QY:0.0000 QZ:0.0000");
  }
  Serial.println(); // End packet
  
  // Stabilize the serial output stream
  delay(20); 
}