# Arduino Nano Quick Reference for Glove Project

## ✅ **Why Arduino Nano is Perfect for This Project**

- **Compact size** - Fits easily on a glove
- **8 analog pins** (A0-A7) - More than enough for 5 flex sensors
- **5V operating voltage** - Standard for flex sensors
- **Low power** - Can run on battery
- **Same as Arduino Uno** - ATmega328P chip, fully compatible

---

## 🔌 **Pinout for 5 Flex Sensors**

```
Arduino Nano
┌─────────────┐
│   USB       │
├─────────────┤
│ D13   D12   │
│ 3V3   D11   │
│ REF   D10   │
│ A0    D9    │  ← Flex 1 (Thumb)
│ A1    D8    │  ← Flex 2 (Index)
│ A2    D7    │  ← Flex 3 (Middle)
│ A3    D6    │  ← Flex 4 (Ring)
│ A4    D5    │  ← Flex 5 (Pinkie)
│ A5    D4    │
│ A6    D3    │
│ A7    D2    │
│ 5V    GND   │
│ RST   RST   │
│ GND   RX0   │
│ VIN   TX1   │
└─────────────┘
```

---

## 🔧 **Wiring Each Flex Sensor (Voltage Divider)**

For each flex sensor:

```
5V ----[Flex Sensor]----+----[10kΩ Resistor]---- GND
                        |
                     Analog Pin (A0-A4)
```

**Example for Thumb (Flex 1):**
- Flex sensor one end → 5V
- Flex sensor other end → A0 AND 10kΩ resistor
- 10kΩ resistor other end → GND

Repeat for all 5 sensors on pins A0-A4.

---

## 💻 **Programming the Nano**

### **1. Arduino IDE Settings:**
- **Board**: Tools → Board → Arduino Nano
- **Processor**: Tools → Processor → ATmega328P (Old Bootloader) *- for most clones*
- **Port**: Tools → Port → COM3 (or your port)

### **2. Upload the Test Sketch:**
```
File → Open → arduino_test_sketch.ino
Click Upload (→)
```

### **3. Open Serial Monitor:**
```
Tools → Serial Monitor
Set baud rate to: 115200
```

---

## 🐛 **Common Arduino Nano Issues**

### **Issue 1: "Port not found" or "Device not recognized"**

**Most likely cause**: Missing CH340 USB driver (for Nano clones)

**Solution:**
1. Download CH340 driver: http://www.wch.cn/downloads/CH341SER_ZIP.html
2. Install the driver
3. Restart computer (or at least replug USB)
4. Check Device Manager → Ports → should see "USB-SERIAL CH340 (COMx)"

---

### **Issue 2: Upload Error "avrdude: stk500_recv(): programmer is not responding"**

**Solution:**
- Change **Processor** to "ATmega328P (Old Bootloader)" in Tools menu
- Try a different USB cable
- Press reset button on Nano right before uploading

---

### **Issue 3: Serial Monitor shows gibberish**

**Solution:**
- Make sure baud rate is set to **115200** (bottom-right of Serial Monitor)
- Check that Arduino sketch uses `Serial.begin(115200)`

---

## 📊 **Expected Sensor Values**

With 10kΩ resistor voltage divider:

| Hand Position | Typical ADC Value | Voltage |
|---------------|-------------------|---------|
| Fully extended | 100-200 | ~0.5-1.0V |
| Relaxed | 300-400 | ~1.5-2.0V |
| Partially bent | 500-700 | ~2.5-3.5V |
| Fully bent | 800-900 | ~4.0-4.5V |

*Values may vary based on your specific flex sensors*

---

## ⚡ **Power Options**

### **Option 1: USB Power (Development)**
- Simply plug USB cable to computer
- Provides 5V, sufficient for 5 flex sensors
- Best for data collection and testing

### **Option 2: Battery Power (Portable)**
- Connect 7-12V battery to VIN and GND pins
- **Recommended**: 9V battery or 2S LiPo (7.4V)
- Nano's onboard regulator converts to 5V
- Use Bluetooth module to send data wirelessly

---

## 📡 **Serial Communication Format**

The Arduino sketch sends data at **50Hz** (20ms interval):

```
Format: flex1,flex2,flex3,flex4,flex5
Example: 123,456,789,012,345
```

This format is directly compatible with:
- `collect_data.py` script
- All training scripts
- Mobile app (with minor adaptation)

---

## 🔋 **Power Consumption Estimate**

- Arduino Nano: ~20mA
- 5 Flex Sensors: ~5mA each = 25mA total
- **Total**: ~45mA at 5V

With a 9V 500mAh battery: ~10 hours of continuous operation

---

## ✅ **Pre-Flight Checklist**

Before collecting data tomorrow:

- [ ] Arduino Nano connected via USB
- [ ] CH340 drivers installed (if needed)
- [ ] All 5 flex sensors wired correctly
- [ ] Test sketch uploaded successfully
- [ ] Serial Monitor shows changing values when moving fingers
- [ ] COM port identified (e.g., COM3)
- [ ] Python environment set up
- [ ] `collect_data.py` can connect to the Nano

---

**You're all set for tomorrow! 🚀**

