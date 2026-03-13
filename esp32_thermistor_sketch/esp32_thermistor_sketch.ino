/*
 * ESP32 Thermistor Glove — Dual Mode (USB Serial + WiFi TCP) + IMU
 *
 * Sends 5-finger thermistor data + BNO055 quaternion at 50 Hz in CSV format:
 *   "thumb,index,middle,ring,pinky,qw,qx,qy,qz\n"
 *
 * Outputs on:
 *   1. USB Serial → desktop app via cable (115200 baud)
 *   2. WiFi TCP   → desktop/mobile app wirelessly
 *
 * WiFi connection (Station mode — joins your existing router):
 *   Set WIFI_SSID / WIFI_PASS below to your home/lab router credentials.
 *   The ESP32 joins that network and gets an IP from DHCP.
 *   mDNS hostname: glove.local  (no need to know the IP)
 *   TCP port     : 3333
 *   → Both ESP32 and laptop are on the same router — laptop keeps internet.
 *   → In the desktop app, connect to glove.local:3333
 *
 * Hardware : ESP32 + 5 thermistors on ADC pins 32, 35, 34, 39, 36
 *            (thumb, index, middle, ring, pinky)
 *            BNO055 IMU: SDA=21, SCL=22, RST=23, ADDR=0x29
 *
 * If BNO055 is not present the sketch falls back to identity quaternion (1,0,0,0)
 * so it still works with the old 9-value parser.
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <WiFi.h>
#include <ESPmDNS.h>

// ── WiFi credentials — change these to your router ───────────────────────────
#define WIFI_SSID  "SoftSensorsLab"
#define WIFI_PASS  "SoftSensors1324?"
#define TCP_PORT   3333
#define MDNS_NAME  "glove"   // reachable at glove.local

// ── IMU pins & address ────────────────────────────────────────────────────────
#define I2C_SDA  21
#define I2C_SCL  22
#define BNO_RST  23
#define BNO_ADDR 0x29

// ── Sensor pins (thumb→pinky) ─────────────────────────────────────────────────
#define NUM_FINGERS 5
const int THERM_PINS[NUM_FINGERS] = {32, 35, 34, 39, 36};

// ── IMU ───────────────────────────────────────────────────────────────────────
Adafruit_BNO055 bno = Adafruit_BNO055(55, BNO_ADDR, &Wire);
bool imuReady = false;

// ── WiFi TCP server ───────────────────────────────────────────────────────────
WiFiServer tcpServer(TCP_PORT);
WiFiClient tcpClient;

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // ── Configure ADC pins ────────────────────────────────────────────────────
  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(THERM_PINS[i], INPUT);
  }

  // ── BNO055 Initialisation ─────────────────────────────────────────────────
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW);  delay(20);
  digitalWrite(BNO_RST, HIGH); delay(100);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(100);

  if (bno.begin()) {
    bno.setExtCrystalUse(true);
    imuReady = true;
    Serial.println("[IMU] BNO055 initialised — quaternion output enabled");
  } else {
    Serial.println("[IMU] BNO055 NOT FOUND — sending identity quaternion");
  }

  // ── WiFi Station mode — join existing router ──────────────────────────────
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("[WiFi] Connecting to ");
  Serial.print(WIFI_SSID);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("[WiFi] Connected! IP: ");
    Serial.println(WiFi.localIP());

    // ── mDNS — advertise as glove.local ──────────────────────────────────
    if (MDNS.begin(MDNS_NAME)) {
      MDNS.addService("tcp", "tcp", TCP_PORT);
      Serial.println("[mDNS] Hostname: " MDNS_NAME ".local");
    } else {
      Serial.println("[mDNS] Failed to start — use IP above instead");
    }

    tcpServer.begin();
    Serial.print("[TCP]  Listening on port ");
    Serial.println(TCP_PORT);
    Serial.println("[READY] Dual-mode: Serial + WiFi TCP (" MDNS_NAME ".local:" + String(TCP_PORT) + ")");
  } else {
    Serial.println();
    Serial.println("[WiFi] Connection FAILED — check SSID/password. Running Serial-only.");
  }
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // ── Accept new TCP client if none is connected ───────────────────────────
  if (!tcpClient || !tcpClient.connected()) {
    WiFiClient newClient = tcpServer.available();
    if (newClient) {
      tcpClient = newClient;
      tcpClient.setNoDelay(true);  // disable Nagle — lower latency at 50 Hz
      Serial.print("[WiFi] Client connected from ");
      Serial.println(tcpClient.remoteIP());
    }
  }

  // ── Read all 5 thermistors ────────────────────────────────────────────────
  int v[NUM_FINGERS];
  for (int i = 0; i < NUM_FINGERS; i++) {
    v[i] = analogRead(THERM_PINS[i]);
  }

  // ── Read IMU quaternion ───────────────────────────────────────────────────
  float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
  if (imuReady) {
    imu::Quaternion quat = bno.getQuat();
    qw = (float)quat.w();
    qx = (float)quat.x();
    qy = (float)quat.y();
    qz = (float)quat.z();
  }

  // ── Build CSV: "thumb,index,middle,ring,pinky,qw,qx,qy,qz\n" ─────────────
  char csv[64];
  snprintf(csv, sizeof(csv), "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
           v[0], v[1], v[2], v[3], v[4], qw, qx, qy, qz);

  // ── Output 1: USB Serial ─────────────────────────────────────────────────
  Serial.print(csv);

  // ── Output 2: WiFi TCP ───────────────────────────────────────────────────
  if (tcpClient && tcpClient.connected()) {
    tcpClient.print(csv);
  }

  delay(20);  // 50 Hz
}
