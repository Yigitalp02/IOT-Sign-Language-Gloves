/*
 * ESP32-C3 Glove — ADS1115 16-bit ADC + BNO055 + WiFi TCP + USB Serial
 *                  + ESP-NOW receiver for armband IMUs (Q1 upper arm, Q2 forearm)
 *
 * Based on prof's hardware revision. Sends 5-finger flex data + BNO055
 * quaternion + linear acceleration + gyroscope at 50 Hz in CSV format.
 *
 * Extended CSV format (23 columns — first 15 UNCHANGED, fully backward compatible):
 *   "thumb,index,middle,ring,pinky,qw,qx,qy,qz,lx,ly,lz,gx,gy,gz,q1w,q1x,q1y,q1z,q2w,q2x,q2y,q2z\n"
 *    col:  0     1      2      3    4   5  6  7  8  9 10 11 12 13 14  15  16  17  18  19  20  21  22
 *
 *   lx,ly,lz  — linear acceleration [m/s²], gravity removed (≈0 when still)
 *   gx,gy,gz  — angular velocity     [deg/s] (wrist rotation speed)
 *   Q1 (cols 15-18) — upper arm / triceps armband (ESP32-S3, id=1)
 *   Q2 (cols 19-22) — forearm armband             (ESP32-S3, id=2)
 *   When armbands are not connected Q1 and Q2 stay at identity (1,0,0,0).
 *
 * Hardware (unchanged from v2):
 *   - Board    : ESP32-C3
 *   - F0 thumb : ESP32-C3 built-in ADC, GPIO 1 (12-bit, 0-4095)
 *   - F1-F4    : ADS1115 external 16-bit ADC via I2C
 *                Channels are board-inverted: F1→A3, F2→A2, F3→A1, F4→A0
 *   - I2C      : SDA=4, SCL=5
 *   - BNO_RST  : GPIO 6
 *
 * WiFi (Station mode — unchanged):
 *   mDNS hostname : glove.local
 *   TCP port      : 3333
 *
 * ── ARMBAND SETUP ─────────────────────────────────────────────────────────────
 *   The two ESP32-S3 armbands send their quaternion via ESP-NOW. They must be
 *   flashed with THIS board's MAC address as their broadcastAddress[].
 *   Find it in Serial Monitor at first boot:
 *     "[ESP-NOW] This board MAC: XX:XX:XX:XX:XX:XX"
 *   Update broadcastAddress[] in both armband .ino files to that MAC,
 *   set id=1 for the upper-arm (triceps) band and id=2 for the forearm band,
 *   then re-flash both armbands.
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <esp_now.h>

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
#define F0_PIN  1

// ── WiFi TCP server ───────────────────────────────────────────────────────────
WiFiServer tcpServer(TCP_PORT);
WiFiClient tcpClient;

// ── ESP-NOW — armband quaternion data ────────────────────────────────────────
// Struct must exactly match the struct_message in both armband .ino files.
typedef struct {
  int   id;   // 1 = upper arm (triceps), 2 = forearm
  float w;
  float x;
  float y;
  float z;
} ArmPacket;

// Latest quaternion from each armband.
// Defaults to identity so columns 15-22 read 1,0,0,0,1,0,0,0 when
// armbands are off — no garbage in the CSV.
// volatile: written by the ESP-NOW WiFi-task callback, read by loop().
volatile float q1w = 1.0f, q1x = 0.0f, q1y = 0.0f, q1z = 0.0f;  // upper arm
volatile float q2w = 1.0f, q2x = 0.0f, q2y = 0.0f, q2z = 0.0f;  // forearm

// Called automatically by the ESP-NOW driver whenever a packet arrives.
// Runs in the WiFi task — keep it short, no Serial prints.
// Signature uses mac_addr (Arduino core 2.x style; core 3.x uses esp_now_recv_info*).
void IRAM_ATTR onArmData(const uint8_t *mac_addr, const uint8_t *data, int len) {
  if (len != sizeof(ArmPacket)) return;   // wrong packet size — ignore
  ArmPacket pkt;
  memcpy(&pkt, data, sizeof(pkt));

  if (pkt.id == 1) {
    q1w = pkt.w;  q1x = pkt.x;  q1y = pkt.y;  q1z = pkt.z;
  } else if (pkt.id == 2) {
    q2w = pkt.w;  q2x = pkt.x;  q2y = pkt.y;  q2z = pkt.z;
  }
}

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
  // WiFi.mode(WIFI_STA) must be called before esp_now_init() so both share
  // the same radio in station mode.
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

  // ── ESP-NOW receiver (armband IMUs) ───────────────────────────────────────
  // Print this board's MAC so you can paste it into the armband firmware.
  Serial.printf("[ESP-NOW] This board MAC: %s\n", WiFi.macAddress().c_str());
  Serial.println("[ESP-NOW] Update broadcastAddress[] in armband .ino files to that MAC.");

  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Init FAILED — armband IMUs will not be received.");
    Serial.println("[ESP-NOW]   Q1/Q2 columns will stay at identity (1,0,0,0).");
  } else {
    esp_now_register_recv_cb(onArmData);
    Serial.println("[ESP-NOW] Ready — waiting for armband packets (id=1 upper arm, id=2 forearm).");
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

  // ── IMU quaternion + linear accel + gyroscope (wrist — Q0) ───────────────
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

  // ── Snapshot armband quaternions (volatile → local, safe single read) ────
  float a1w = q1w, a1x = q1x, a1y = q1y, a1z = q1z;  // upper arm
  float a2w = q2w, a2x = q2x, a2y = q2y, a2z = q2z;  // forearm

  // ── Build CSV ─────────────────────────────────────────────────────────────
  // Cols 0-14: UNCHANGED — existing desktop/mobile apps read only these.
  // Cols 15-22: NEW — arm IMUs; identity when armbands are off.
  //
  // "thumb,index,middle,ring,pinky,qw,qx,qy,qz,lx,ly,lz,gx,gy,gz,q1w,q1x,q1y,q1z,q2w,q2x,q2y,q2z\n"
  char csv[256];
  snprintf(csv, sizeof(csv),
           "%d,%d,%d,%d,%d,"          // cols  0-4  flex sensors
           "%.4f,%.4f,%.4f,%.4f,"     // cols  5-8  wrist quaternion Q0
           "%.3f,%.3f,%.3f,"          // cols  9-11 linear accel
           "%.2f,%.2f,%.2f,"          // cols 12-14 gyroscope
           "%.4f,%.4f,%.4f,%.4f,"     // cols 15-18 upper arm Q1
           "%.4f,%.4f,%.4f,%.4f\n",   // cols 19-22 forearm Q2
           v[0], v[1], v[2], v[3], v[4],
           qw, qx, qy, qz,
           lx, ly, lz,
           gx, gy, gz,
           a1w, a1x, a1y, a1z,
           a2w, a2x, a2y, a2z);

  // ── Output 1: USB Serial ─────────────────────────────────────────────────
  Serial.print(csv);

  // ── Output 2: WiFi TCP ───────────────────────────────────────────────────
  if (tcpClient && tcpClient.connected()) {
    tcpClient.print(csv);
  }

  delay(20);  // 50 Hz
}
