/*
 * ESP32-C3 Glove — ADS1115 16-bit ADC + BNO055 + WiFi TCP + USB Serial
 *
 * Based on prof's hardware revision. Sends 5-finger flex data + BNO055
 * quaternion + linear acceleration + gyroscope at 50 Hz in CSV format.
 * Existing apps only read the first 9 columns so they need no changes.
 *   "thumb,index,middle,ring,pinky,qw,qx,qy,qz,lx,ly,lz,gx,gy,gz\n"
 *    col:  0     1      2      3    4   5  6  7  8  9 10 11 12 13 14
 *
 *   lx,ly,lz — linear acceleration [m/s²], gravity removed (≈0 when still)
 *   gx,gy,gz — angular velocity     [deg/s] (wrist rotation speed)
 *
 * Hardware changes from v1 (esp32_thermistor_sketch):
 *   - Board    : ESP32-C3 (instead of standard ESP32)
 *   - F0 thumb : ESP32-C3 built-in ADC, GPIO 0 (12-bit, 0-4095)
 *   - F1-F4    : ADS1115 external 16-bit ADC via I2C
 *                Channels are board-inverted: F1→A3, F2→A2, F3→A1, F4→A0
 *                Full 16-bit values sent (0-~26000 at 3.3V with GAIN_ONE)
 *                NOTE: Per-finger calibration in the app handles the
 *                      different scale between F0 (12-bit) and F1-F4 (16-bit).
 *   - I2C      : SDA=4, SCL=5
 *   - BNO_RST  : GPIO 6
 *
 * WiFi (Station mode — same as v1):
 *   Set WIFI_SSID / WIFI_PASS to your router credentials.
 *   mDNS hostname : glove.local
 *   TCP port      : 3333
 *   → Connect in the app with: glove.local:3333
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <ESPmDNS.h>

// ── WiFi credentials — change these to your router ───────────────────────────
#define WIFI_SSID  "SoftSensorsLab"
#define WIFI_PASS  "SoftSensors1324?"
#define TCP_PORT   3333
#define MDNS_NAME  "glove"   // reachable as glove.local

// ── I2C pins (ESP32-C3) ───────────────────────────────────────────────────────
#define I2C_SDA  4
#define I2C_SCL  5

// ── BNO055 ────────────────────────────────────────────────────────────────────
#define BNO_RST  6
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);
bool imuReady = false;

// ── ADS1115 (F1-F4) ───────────────────────────────────────────────────────────
Adafruit_ADS1115 ads;
bool adsReady = false;

// ── F0 pin — ESP32-C3 built-in 12-bit ADC (thumb) ────────────────────────────
#define F0_PIN  0

// ── WiFi TCP server ───────────────────────────────────────────────────────────
WiFiServer tcpServer(TCP_PORT);
WiFiClient tcpClient;

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(100); // brief settle — don't block on Serial (ESP32-C3 USB CDC would stall without a host)

  // ── Reset BNO055 hardware ─────────────────────────────────────────────────
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW);  delay(20);
  digitalWrite(BNO_RST, HIGH); delay(100);

  // ── I2C bus ───────────────────────────────────────────────────────────────
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(100);

  // ── I2C scanner — detect which devices are present ───────────────────────
  Serial.println("[I2C] Scanning bus...");
  bool bnoFound = false, adsFound = false;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[I2C] Found device at 0x%02X\n", addr);
      if (addr == 0x29) bnoFound = true;
      if (addr == 0x48) adsFound = true;
    }
  }

  // ── BNO055 ───────────────────────────────────────────────────────────────
  if (bnoFound && bno.begin()) {
    bno.setExtCrystalUse(true);
    imuReady = true;
    Serial.println("[IMU] BNO055 initialised — quaternion + linear accel + gyroscope enabled");
  } else {
    Serial.println("[IMU] BNO055 NOT FOUND — sending identity quaternion (1,0,0,0)");
  }

  // ── ADS1115 ──────────────────────────────────────────────────────────────
  if (adsFound && ads.begin()) {
    // GAIN_ONE = ±4.096V range — optimal for 3.3V flex-sensor dividers;
    // gives 1 LSB = 0.125 mV and ~26400 counts full-scale at 3.3V.
    ads.setGain(GAIN_ONE);
    // 475 SPS: 4 channels × ~2.1 ms = ~8.4 ms per loop cycle — well under 20 ms
    ads.setDataRate(RATE_ADS1115_475SPS);
    adsReady = true;
    Serial.println("[ADC] ADS1115 initialised — F1-F4 at full 16-bit precision");
  } else {
    Serial.println("[ADC] ADS1115 NOT FOUND — F1-F4 will output 0");
  }

  // ── F0 ADC (ESP32-C3 built-in) ────────────────────────────────────────────
  pinMode(F0_PIN, INPUT);
  Serial.println("[ADC] F0 (thumb) on ESP32-C3 ADC GPIO " + String(F0_PIN) + " (12-bit)");

  // ── WiFi Station mode ─────────────────────────────────────────────────────
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting to " WIFI_SSID);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

    if (MDNS.begin(MDNS_NAME)) {
      MDNS.addService("tcp", "tcp", TCP_PORT);
      Serial.println("[mDNS] Hostname: " MDNS_NAME ".local");
    } else {
      Serial.println("[mDNS] Failed to start — use IP above instead");
    }

    tcpServer.begin();
    Serial.printf("[TCP]  Listening on port %d\n", TCP_PORT);
    Serial.println("[READY] Dual-mode: Serial + WiFi TCP (" MDNS_NAME ".local:" + String(TCP_PORT) + ")");
  } else {
    Serial.println("\n[WiFi] Connection FAILED — check SSID/password. Running Serial-only.");
  }
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // ── Accept new TCP client if none connected ───────────────────────────────
  if (!tcpClient || !tcpClient.connected()) {
    WiFiClient nc = tcpServer.available();
    if (nc) {
      tcpClient = nc;
      tcpClient.setNoDelay(true);  // disable Nagle — lower latency at 50 Hz
      Serial.printf("[WiFi] Client connected from %s\n", tcpClient.remoteIP().toString().c_str());
    }
  }

  int v[5];

  // ── F0 (thumb) — ESP32-C3 built-in ADC, 12-bit (0-4095) ─────────────────
  v[0] = analogRead(F0_PIN);

  // ── F1-F4 — ADS1115, full 16-bit (0-~26000 typical at 3.3V / GAIN_ONE) ──
  // Board channels are inverted: finger 1 → A3, finger 2 → A2, etc.
  for (int i = 1; i < 5; i++) {
    v[i] = adsReady ? ads.readADC_SingleEnded(4 - i) : 0;
  }

  // ── IMU quaternion + linear accel + gyroscope ────────────────────────────
  float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
  float lx = 0.0f, ly = 0.0f, lz = 0.0f;  // linear acceleration [m/s²]
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;  // angular velocity    [deg/s]
  if (imuReady) {
    imu::Quaternion q = bno.getQuat();
    qw = (float)q.w();
    qx = (float)q.x();
    qy = (float)q.y();
    qz = (float)q.z();

    imu::Vector<3> la = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
    lx = (float)la.x();
    ly = (float)la.y();
    lz = (float)la.z();

    imu::Vector<3> gy3 = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    gx = (float)gy3.x();
    gy = (float)gy3.y();
    gz = (float)gy3.z();
  }

  // ── Build CSV ─────────────────────────────────────────────────────────────
  // "thumb,index,middle,ring,pinky,qw,qx,qy,qz,lx,ly,lz,gx,gy,gz\n"
  char csv[128];
  snprintf(csv, sizeof(csv),
           "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
           v[0], v[1], v[2], v[3], v[4],
           qw, qx, qy, qz,
           lx, ly, lz,
           gx, gy, gz);

  // ── Output 1: USB Serial ─────────────────────────────────────────────────
  Serial.print(csv);

  // ── Output 2: WiFi TCP ───────────────────────────────────────────────────
  if (tcpClient && tcpClient.connected()) {
    tcpClient.print(csv);
  }

  delay(20);  // 50 Hz
}
