# 🧪 Testing Your Glove Connection

This guide helps you test your glove's serial connection before collecting data.

---

## 🎯 Goal

Verify that:
1. Your glove is connected and recognized by the computer
2. Sensor data is streaming correctly
3. All 5 flex sensors are working
4. Data format is compatible with the collection script

---

## 🔌 **Step 1: Physical Connection**

1. Connect your **Arduino Nano** glove to your computer via USB
2. Wait for the device to be recognized (you might hear a sound on Windows)
3. Note the COM port (Windows) or device name (Linux/Mac)

### **⚠️ Arduino Nano Driver Note (Windows)**

Many Arduino Nano clones use the **CH340** USB-to-Serial chip instead of FTDI. If Windows doesn't recognize your Nano:

1. **Download CH340 drivers**: [http://www.wch.cn/downloads/CH341SER_ZIP.html](http://www.wch.cn/downloads/CH341SER_ZIP.html)
2. Extract and run `CH341SER.EXE`
3. Click "Install"
4. Unplug and replug your Arduino Nano
5. Check Device Manager - it should now appear as "USB-SERIAL CH340 (COMx)"

### **Find the COM Port**

#### **Windows:**
```powershell
# Method 1: Device Manager
# Open Device Manager → Ports (COM & LPT) → Look for:
#   - "USB-SERIAL CH340 (COMx)" - for Nano clones with CH340
#   - "Arduino Nano (COMx)" - for official Nano with FTDI
#   - "USB Serial Device (COMx)" - generic

# Method 2: PowerShell
Get-WmiObject -Class Win32_SerialPort | Select-Object Name, DeviceID
```

#### **Linux/Mac:**
```bash
# List USB serial devices
ls /dev/tty*

# Common patterns:
# - Linux: /dev/ttyUSB0, /dev/ttyACM0
# - Mac: /dev/tty.usbserial*, /dev/tty.usbmodem*
```

---

## 📡 **Step 2: Test with Arduino Serial Monitor**

### **Option A: Using Arduino IDE**

1. Open Arduino IDE
2. Go to **Tools → Port** and select your glove's port
3. Upload the test sketch: `arduino_test_sketch.ino`
4. Open **Tools → Serial Monitor**
5. Set baud rate to **115200**

**Expected output:**
```
123,456,789,012,345
124,455,788,013,346
122,457,790,011,344
...
```

### **Option B: Using PuTTY (Windows)**

1. Download PuTTY if you don't have it
2. Open PuTTY
3. Connection Type: **Serial**
4. Serial Line: **COM3** (or your port)
5. Speed: **115200**
6. Click **Open**

### **Option C: Using screen (Linux/Mac)**

```bash
screen /dev/ttyUSB0 115200

# To exit: Ctrl+A, then K, then Y
```

---

## 🔍 **Step 3: Verify Sensor Data**

### **What to Check:**

1. **Data is streaming** - New lines appear continuously
2. **Format is correct** - Five comma-separated numbers
3. **Values change** - Move your fingers and verify numbers change
4. **Range is reasonable** - Values typically 0-1023 (analog read range)

### **Test Each Finger:**

| Finger | Sensor | Test Action | Expected Change |
|--------|--------|-------------|-----------------|
| Thumb  | flex_1 | Bend thumb | Value increases |
| Index  | flex_2 | Bend index | Value increases |
| Middle | flex_3 | Bend middle | Value increases |
| Ring   | flex_4 | Bend ring | Value increases |
| Pinkie | flex_5 | Bend pinkie | Value increases |

### **Example Test Sequence:**

1. **Open hand** - Record baseline values
   ```
   120,180,200,210,190
   ```

2. **Make fist** - All values should increase
   ```
   580,720,770,810,690
   ```

3. **Bend only index finger** - Only flex_2 should change
   ```
   120,720,200,210,190
   ```

---

## 🐍 **Step 4: Test with Python Script**

Create a simple test script:

```python
import serial
import serial.tools.list_ports

# List available ports
print("Available ports:")
for port in serial.tools.list_ports.comports():
    print(f"  - {port.device}: {port.description}")

# Connect to your glove (update COM port)
PORT = 'COM3'  # Change this to your port
BAUDRATE = 115200

print(f"\nConnecting to {PORT}...")
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# Wait for Arduino to reset
import time
time.sleep(2)

print("Reading data (Ctrl+C to stop):\n")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"  {line}")
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    ser.close()
    print("Connection closed")
```

Save as `test_serial.py` and run:
```powershell
python test_serial.py
```

---

## ✅ **Step 5: Test with Data Collection Script**

Once serial connection is verified:

```powershell
cd C:\Users\Yigit\Desktop\iot-sign-language-desktop\iot-sign-glove
python scripts/collect_data.py
```

Choose option **1** (Test connection) and verify live sensor data.

---

## 🐛 **Troubleshooting**

### **Problem: No ports detected**

**Possible causes:**
- USB cable is charge-only (no data)
- Arduino drivers not installed
- Arduino not powered on

**Solutions:**
- Try a different USB cable (must support data, not just power)
- **For Arduino Nano clones**: Install CH340 drivers (see Step 1 above)
- **For official Arduino Nano**: Install FTDI drivers
- Check Device Manager for "Unknown Device" or devices with yellow warning icon
- Try a different USB port (preferably USB 2.0, not USB 3.0)

---

### **Problem: "Access denied" or "Port already in use"**

**Possible causes:**
- Another program is using the port (Arduino IDE, PuTTY, etc.)
- Previous connection not closed properly

**Solutions:**
- Close all programs that might be using the port
- Unplug and replug the USB cable
- Restart your computer

---

### **Problem: Garbage/corrupted data**

**Possible causes:**
- Wrong baud rate
- Timing issues
- Buffer overflow

**Solutions:**
- Verify baud rate is 115200 in both Arduino and Python
- Add delay in Arduino sketch between samples
- Add `time.sleep(0.01)` in Python read loop

---

### **Problem: Sensor values not changing**

**Possible causes:**
- Sensor not connected
- Wrong analog pin
- Sensor damaged

**Solutions:**
- Check wiring connections
- Verify analog pin numbers in Arduino code
- Test each sensor individually with a multimeter
- Try swapping sensors to isolate the issue

---

### **Problem: Values always at 0 or 1023**

**Possible causes:**
- Voltage divider circuit issue
- Missing pull-up/pull-down resistor
- Sensor short circuit

**Solutions:**
- Check voltage divider resistor values (should be ~10KΩ)
- Verify 5V and GND connections
- Measure voltage at analog pin with multimeter (should be 0-5V)

---

## 📊 **Expected Sensor Ranges**

Typical flex sensor values (with 10K resistor):

| Hand Position | Flex Value Range |
|---------------|------------------|
| Fully extended | 100-200 |
| Relaxed | 300-400 |
| Partially bent | 500-700 |
| Fully bent | 800-900 |

*Note: Actual values depend on your specific sensors and circuit.*

---

## ✅ **Success Criteria**

Your glove is ready for data collection when:
- ✅ Serial port is detected
- ✅ Data streams continuously at ~50Hz
- ✅ Format matches expected: `flex1,flex2,flex3,flex4,flex5`
- ✅ All 5 sensors respond to finger movements
- ✅ Values are in reasonable range (0-1023)
- ✅ No error messages or corrupt data

---

## 🎯 **Next Steps**

Once your connection test passes:

1. ✅ Run full data collection: `python scripts/collect_data.py`
2. ✅ Collect 5-10 repetitions of each letter
3. ✅ Validate data: `python scripts/validate_data.py --data data/my_glove_data`
4. ✅ Train model: `python scripts/train_model.py --data data/my_glove_data`

**Good luck! 🍀**

