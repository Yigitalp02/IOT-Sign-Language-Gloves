/*
 * ESP32 Thermistor Glove — Dual Mode (USB Serial + BLE UART) + IMU
 *
 * Sends 5-finger thermistor data + BNO055 quaternion at 50 Hz in CSV format:
 *   "thumb,index,middle,ring,pinky,qw,qx,qy,qz\n"
 *
 * Outputs on:
 *   1. USB Serial    → desktop app (115200 baud)
 *   2. BLE UART/NUS  → mobile app (Nordic UART Service)
 *
 * BLE device name : ESP32-GloveASL
 * Service UUID    : 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 * TX char UUID    : 6E400003-B5A3-F393-E0A9-E50E24DCCA9E  (ESP32 → phone)
 * RX char UUID    : 6E400002-B5A3-F393-E0A9-E50E24DCCA9E  (phone → ESP32, reserved)
 *
 * Hardware : ESP32 + 5 thermistors on ADC pins 32, 35, 34, 39, 36
 *            (thumb, index, middle, ring, pinky)
 *            BNO055 IMU: SDA=21, SCL=22, RST=23, ADDR=0x29
 *
 * If BNO055 is not present the sketch falls back to identity quaternion (0,0,0,1)
 * so it still works with the old 9-value parser.
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── BLE UUIDs (Nordic UART Service) ──────────────────────────────────────────
#define NUS_SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_CHAR_TX_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // notify
#define NUS_CHAR_RX_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // write

// ── IMU pins & address ────────────────────────────────────────────────────────
#define I2C_SDA   21
#define I2C_SCL   22
#define BNO_RST   23       // Reset pin (active LOW)
#define BNO_ADDR  0x29     // AD0 pin tied HIGH on this board

// ── Sensor pins (thumb→pinky) ─────────────────────────────────────────────────
#define NUM_FINGERS 5
const int THERM_PINS[NUM_FINGERS] = {32, 35, 34, 39, 36};

// ── IMU ───────────────────────────────────────────────────────────────────────
Adafruit_BNO055 bno = Adafruit_BNO055(55, BNO_ADDR, &Wire);
bool imuReady = false;

// ── BLE state ─────────────────────────────────────────────────────────────────
BLEServer         *pServer           = nullptr;
BLECharacteristic *pTxCharacteristic = nullptr;
volatile bool      bleConnected      = false;
volatile bool      bleWasConnected   = false;

class GloveServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override {
    bleConnected = true;
    BLEDevice::stopAdvertising();
    Serial.println("[BLE] Client connected");
  }
  void onDisconnect(BLEServer *) override {
    bleConnected    = false;
    bleWasConnected = true;
    Serial.println("[BLE] Client disconnected");
  }
};

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // ── Configure ADC pins ────────────────────────────────────────────────────
  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(THERM_PINS[i], INPUT);
  }

  // ── BNO055 Initialisation ─────────────────────────────────────────────────
  // Hardware reset
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW);  delay(20);
  digitalWrite(BNO_RST, HIGH); delay(100);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(100);  // Critical fix for ESP32 I2C clock stretching

  if (bno.begin()) {
    bno.setExtCrystalUse(true);
    imuReady = true;
    Serial.println("[IMU] BNO055 initialised — quaternion output enabled");
  } else {
    Serial.println("[IMU] BNO055 NOT FOUND — sending identity quaternion (0,0,0,1)");
  }

  // ── BLE initialisation ────────────────────────────────────────────────────
  BLEDevice::init("ESP32-GloveASL");
  BLEDevice::setMTU(128);  // Extended MTU for larger 9-value packets

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new GloveServerCallbacks());

  BLEService *pService = pServer->createService(NUS_SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
    NUS_CHAR_TX_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pTxCharacteristic->addDescriptor(new BLE2902());

  pService->createCharacteristic(
    NUS_CHAR_RX_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );

  pService->start();

  BLEAdvertising *pAdv = BLEDevice::getAdvertising();
  pAdv->addServiceUUID(NUS_SERVICE_UUID);
  pAdv->setScanResponse(true);
  pAdv->setMinPreferred(0x06);
  pAdv->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("[READY] ESP32 Glove — USB Serial + BLE advertising (9-value mode)");
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // Restart BLE advertising after disconnect
  if (bleWasConnected && !bleConnected) {
    bleWasConnected = false;
    delay(100);
    BLEDevice::startAdvertising();
    Serial.println("[BLE] Restarted advertising");
  }

  // ── Read all 5 thermistors ────────────────────────────────────────────────
  int v[NUM_FINGERS];
  for (int i = 0; i < NUM_FINGERS; i++) {
    v[i] = analogRead(THERM_PINS[i]);
  }

  // ── Read IMU quaternion (or use identity if unavailable) ──────────────────
  float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;  // identity = no rotation
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

  // ── USB Serial → desktop app ──────────────────────────────────────────────
  Serial.print(csv);

  // ── BLE notify → mobile app ───────────────────────────────────────────────
  if (bleConnected) {
    pTxCharacteristic->setValue(reinterpret_cast<uint8_t *>(csv), strlen(csv));
    pTxCharacteristic->notify();
  }

  delay(20);  // 50 Hz
}
