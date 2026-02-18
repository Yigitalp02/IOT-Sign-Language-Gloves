/*
Arduino Test Sketch for IoT Sign Language Glove
================================================

This is a simple test sketch to verify your glove's serial communication.

Upload this to your Arduino Nano, open the Serial Monitor (115200 baud),
and verify you see sensor values streaming.

Expected output format: flex1,flex2,flex3,flex4,flex5

Hardware:
- Arduino Nano (ATmega328P)
- 5 flex sensors connected to analog pins A0-A4
- Each sensor in a voltage divider circuit with a 10K resistor

Note: If using a Nano clone with CH340 USB chip, you may need to install
      CH340 drivers: http://www.wch.cn/downloads/CH341SER_ZIP.html
*/

// Flex sensor pins
const int FLEX_1 = A0;  // Thumb
const int FLEX_2 = A1;  // Index
const int FLEX_3 = A2;  // Middle
const int FLEX_4 = A3;  // Ring
const int FLEX_5 = A4;  // Pinkie

// Sample rate
const int SAMPLE_DELAY_MS = 20;  // 50Hz (20ms between samples)

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Wait for serial to be ready
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("IoT Sign Language Glove - Test Mode");
  Serial.println("====================================");
  Serial.println("Flex sensor values will stream at 50Hz");
  Serial.println("Expected format: flex1,flex2,flex3,flex4,flex5");
  Serial.println("====================================");
  delay(2000);
}

void loop() {
  // Read all 5 flex sensors
  int flex1 = analogRead(FLEX_1);
  int flex2 = analogRead(FLEX_2);
  int flex3 = analogRead(FLEX_3);
  int flex4 = analogRead(FLEX_4);
  int flex5 = analogRead(FLEX_5);
  
  // Send as CSV format (easier to parse)
  Serial.print(flex1);
  Serial.print(",");
  Serial.print(flex2);
  Serial.print(",");
  Serial.print(flex3);
  Serial.print(",");
  Serial.print(flex4);
  Serial.print(",");
  Serial.println(flex5);
  
  // Optional: Use start/end markers for more robust parsing
  // Format: !flex1,flex2,flex3,flex4,flex5#
  // Uncomment the following and comment out the CSV section above:
  /*
  Serial.print("!");
  Serial.print(flex1);
  Serial.print(",");
  Serial.print(flex2);
  Serial.print(",");
  Serial.print(flex3);
  Serial.print(",");
  Serial.print(flex4);
  Serial.print(",");
  Serial.print(flex5);
  Serial.println("#");
  */
  
  // Wait before next sample (50Hz = 20ms)
  delay(SAMPLE_DELAY_MS);
}

