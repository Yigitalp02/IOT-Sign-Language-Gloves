/*
 * BNO055 Full Capability Diagnostic
 * Upload this sketch, open Serial Monitor at 115200 baud.
 * Prints ALL data vectors the BNO055 can produce at ~5 Hz so you
 * can read comfortably and see what each channel does live.
 *
 * Channels printed:
 *   CAL   : calibration status (0=uncalibrated … 3=fully calibrated)
 *            sys / gyro / accel / mag  — all must reach 3 for best accuracy
 *   EULER : fused heading (yaw) / roll / pitch  [degrees]
 *   QUAT  : fused orientation quaternion  (what we currently stream)
 *   LACC  : LINEAR ACCELERATION — hand movement with gravity subtracted [m/s²]
 *            → THIS is what "if it moves around" means; non-zero when
 *              you translate the glove up/down/left/right/forward/back
 *   GYRO  : raw angular velocity — how fast the wrist is rotating [rad/s or deg/s]
 *   GRAV  : gravity vector direction (tells which way is down)  [m/s²]
 *   ACCEL : raw accelerometer (gravity + motion combined)  [m/s²]
 *   MAG   : raw magnetometer (compass direction)  [µT]
 *   TEMP  : board temperature  [°C]
 *
 * Hardware (ESP32-C3 with prof's board):
 *   SDA = 4, SCL = 5, BNO_RST = 6, I2C address = 0x29
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#define I2C_SDA  4
#define I2C_SCL  5
#define BNO_RST  6

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);

void setup() {
  Serial.begin(115200);
  delay(200);

  // Hardware reset
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW);  delay(20);
  digitalWrite(BNO_RST, HIGH); delay(150);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  if (!bno.begin()) {
    Serial.println("ERROR: BNO055 not found. Check wiring.");
    while (1) delay(1000);
  }
  bno.setExtCrystalUse(true);

  Serial.println("=== BNO055 Full Diagnostic ===");
  Serial.println("Move your hand and watch how each channel responds.\n");
  delay(500);
}

void loop() {
  // ── Calibration status ──────────────────────────────────────────────────
  uint8_t sys, gyro, accel, mag;
  bno.getCalibration(&sys, &gyro, &accel, &mag);
  Serial.printf("CAL   sys=%d gyro=%d accel=%d mag=%d%s\n",
                sys, gyro, accel, mag,
                (sys == 3 && gyro == 3 && accel == 3) ? "  [FULLY CALIBRATED]" : "");

  // ── Euler angles (fused) ────────────────────────────────────────────────
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  Serial.printf("EULER heading=%.1f  roll=%.1f  pitch=%.1f  [deg]\n",
                euler.x(), euler.y(), euler.z());

  // ── Quaternion (what we currently stream) ───────────────────────────────
  imu::Quaternion q = bno.getQuat();
  Serial.printf("QUAT  w=%.4f  x=%.4f  y=%.4f  z=%.4f\n",
                q.w(), q.x(), q.y(), q.z());

  // ── Linear acceleration — movement without gravity ──────────────────────
  // Non-zero ONLY when the hand actually moves through space.
  // Stays near (0,0,0) when hand is still (even tilted).
  imu::Vector<3> lacc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  Serial.printf("LACC  x=%.3f  y=%.3f  z=%.3f  [m/s2]  <- hand translation\n",
                lacc.x(), lacc.y(), lacc.z());

  // ── Gyroscope — rotation rate ────────────────────────────────────────────
  imu::Vector<3> gyrov = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  Serial.printf("GYRO  x=%.2f  y=%.2f  z=%.2f  [deg/s]  <- wrist spin speed\n",
                gyrov.x(), gyrov.y(), gyrov.z());

  // ── Gravity vector ───────────────────────────────────────────────────────
  imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  Serial.printf("GRAV  x=%.3f  y=%.3f  z=%.3f  [m/s2]  <- which way is down\n",
                grav.x(), grav.y(), grav.z());

  // ── Raw accelerometer (gravity + motion) ─────────────────────────────────
  imu::Vector<3> accelv = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  Serial.printf("ACCEL x=%.3f  y=%.3f  z=%.3f  [m/s2]  (GRAV + LACC)\n",
                accelv.x(), accelv.y(), accelv.z());

  // ── Magnetometer ─────────────────────────────────────────────────────────
  imu::Vector<3> mag3 = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
  Serial.printf("MAG   x=%.1f  y=%.1f  z=%.1f  [uT]  <- compass\n",
                mag3.x(), mag3.y(), mag3.z());

  // ── Temperature ──────────────────────────────────────────────────────────
  Serial.printf("TEMP  %d C\n", bno.getTemp());

  Serial.println("--------------------------------------------------");
  delay(200); // 5 Hz — slow enough to read comfortably
}
