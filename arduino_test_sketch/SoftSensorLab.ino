#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/* * BNO055 QUATERNION & FINGER SENSOR CODE
 * This outputs raw Quaternions (QW, QX, QY, QZ) to prevent Gimbal Lock/Jitter.
 */

// --- BNO055 CONFIG ---
#define I2C_SDA 21
#define I2C_SCL 22
#define BNO_RST 23  // CHANGED: Moved from 18 to 23 (18 is used by MOSFET)

// ID 55, address 0x28
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);

// --- FINGER SENSOR CONFIG ---
#define NUM_FINGERS 5
const int thermPins[NUM_FINGERS] = {36, 39, 34, 35, 32};
const int mosfetPins[NUM_FINGERS] = {18, 5, 17, 16, 4};

// Control Logic Variables
int low_temp_target = 1400;
const int tempRange = 200;

// PWM Settings (ESP32 specific)
const int pwmFreq = 5000;
const int pwmResolution = 8; 

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // 1. Setup BNO055 with Reset & Timeout Fixes
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW); delay(20);
  digitalWrite(BNO_RST, HIGH); delay(100);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000); 
  Wire.setTimeOut(100); // Critical fix for ESP32 clock stretching

  if (!bno.begin()) {
    Serial.println("Error: BNO055 not detected! Check wiring.");
    while(1);
  }
  bno.setExtCrystalUse(true);

  // 2. Setup Fingers (PWM)
  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(mosfetPins[i], OUTPUT);
    ledcAttach(mosfetPins[i], pwmFreq, pwmResolution);
    ledcWrite(mosfetPins[i], 0); 
  }
}

void loop() {
  // --- READ QUATERNIONS (No Gimbal Lock) ---
  imu::Quaternion quat = bno.getQuat();

  // Print format: QW:value QX:value ...
  Serial.print("QW:"); Serial.print(quat.w(), 4);
  Serial.print(" QX:"); Serial.print(quat.x(), 4);
  Serial.print(" QY:"); Serial.print(quat.y(), 4);
  Serial.print(" QZ:"); Serial.print(quat.z(), 4);

  // --- READ FINGERS & CONTROL MOSFETS ---
  for (int i = 0; i < NUM_FINGERS; i++) {
    int raw = analogRead(thermPins[i]);
    int duty = 0;

    // Control Logic
    if (raw >= low_temp_target + tempRange) {
      duty = (1 << pwmResolution) - 1; 
    } else if (raw > low_temp_target) {
      duty = map(raw, low_temp_target, low_temp_target + tempRange, 0, (1 << pwmResolution) - 1);
    } else {
      duty = 0;
    }

    ledcWrite(mosfetPins[i], duty);

    // Append Finger Data
    Serial.print(" F"); Serial.print(i); Serial.print(":"); Serial.print(raw);
  }

  Serial.println(); // End packet
  delay(30); // 30ms = ~33 Updates per second (Smooth)
}