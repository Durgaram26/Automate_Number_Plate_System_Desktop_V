#include <Preferences.h>

// ==== Pin Configuration ====
// A4988 Stepper Motor Driver
#define STEP_PIN 25
#define DIR_PIN 26
#define ENABLE_PIN -1   // set to the pin connected to A4988/DRV8825 ENABLE (LOW=enable). Keep -1 if not wired.
// Direction invert: set true to reverse CW/CCW due to wiring/mechanics
const bool INVERT_DIRECTION = true;

// ==== Stepper Config ====
const int stepsFor90Deg = 50;         // 50 full steps = 90°
const int moveDurationMs = 2000;      // 2 seconds to move 90°
const int stepDelayMicros = (moveDurationMs * 1000) / stepsFor90Deg / 2; // half-period high/low

// ==== State Handling ====
enum State { IDLE, WAITING_AT_TOP };
State currentState = IDLE;
unsigned long stateStartTime = 0;
int currentAngle = 0;  // 0 or 90

// Auto close after delay (ms)
const unsigned long holdAtTopMs = 2000;

Preferences preferences;

void enableDriver() {
  if (ENABLE_PIN >= 0) {
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW); // LOW to enable for A4988/DRV8825
  }
}

void disableDriver() {
  if (ENABLE_PIN >= 0) {
    digitalWrite(ENABLE_PIN, HIGH); // HIGH disables on A4988/DRV8825
  }
}

void rotateStepper(bool clockwise, int steps, int delayMicros) {
  digitalWrite(DIR_PIN, (clockwise ^ INVERT_DIRECTION) ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH); delayMicroseconds(delayMicros);
    digitalWrite(STEP_PIN, LOW); delayMicroseconds(delayMicros);
  }
  Serial.print("MOVED steps=");
  Serial.print(steps);
  Serial.print(" dir=");
  Serial.println(clockwise ? "CW" : "CCW");
}

void openGate() {
  if (currentAngle == 90) return;
  enableDriver();
  rotateStepper(true, stepsFor90Deg, stepDelayMicros);
  currentAngle = 90;
  preferences.putInt("angle", currentAngle);
  currentState = WAITING_AT_TOP;
  stateStartTime = millis();
  Serial.println("OK OPENED");
}

void closeGate() {
  if (currentAngle == 0) return;
  enableDriver();
  rotateStepper(false, stepsFor90Deg, stepDelayMicros);
  currentAngle = 0;
  preferences.putInt("angle", currentAngle);
  currentState = IDLE;
  Serial.println("OK CLOSED");
}

void reportStatus() {
  Serial.print("STATUS angle=");
  Serial.print(currentAngle);
  Serial.print(" state=");
  Serial.println(currentState == IDLE ? "IDLE" : "WAITING_AT_TOP");
}

void printHelp() {
  Serial.println("Commands: OPEN | CLOSE | STATUS | PING | TEST | HELP");
}

void testMotion() {
  enableDriver();
  rotateStepper(true, 10, 500);
  delay(100);
  rotateStepper(false, 10, 500);
  Serial.println("OK TESTED");
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  Serial.print("CMD "); Serial.println(cmd);
  if (cmd == "OPEN") {
    openGate();
  } else if (cmd == "CLOSE") {
    closeGate();
  } else if (cmd == "STATUS") {
    reportStatus();
  } else if (cmd == "PING") {
    Serial.println("PONG");
  } else if (cmd == "TEST") {
    testMotion();
  } else if (cmd == "HELP") {
    printHelp();
  } else if (cmd.length() > 0) {
    Serial.println("ERR UNKNOWN_CMD");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(STEP_PIN, OUTPUT); digitalWrite(STEP_PIN, LOW);
  pinMode(DIR_PIN, OUTPUT); digitalWrite(DIR_PIN, LOW);
  if (ENABLE_PIN >= 0) {
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW);
  }

  // ==== EEPROM Initialization ====
  preferences.begin("stepper", false);
  currentAngle = preferences.getInt("angle", 0);
  Serial.print("Recovered angle from memory: ");
  Serial.println(currentAngle);
  Serial.print("Config: STEP="); Serial.print(STEP_PIN);
  Serial.print(" DIR="); Serial.print(DIR_PIN);
  Serial.print(" EN="); Serial.print(ENABLE_PIN);
  Serial.print(" stepsFor90="); Serial.print(stepsFor90Deg);
  Serial.print(" stepDelay(us)="); Serial.println(stepDelayMicros);

  // ==== Homing if needed ====
  if (currentAngle == 90) {
    Serial.println("Restoring to rest position (0°)...");
    rotateStepper(false, stepsFor90Deg, stepDelayMicros);
    currentAngle = 0;
    preferences.putInt("angle", currentAngle); // Save updated position
  }

  Serial.println("System Ready. Send OPEN/CLOSE/STATUS or HELP");
}

void loop() {
  // Handle serial input commands
  static String input;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    Serial.write(c); // echo
    if (c == '\n' || c == '\r') {
      handleCommand(input);
      input = "";
    } else {
      input += c;
    }
  }

  // Auto close after holdAtTopMs
  if (currentState == WAITING_AT_TOP) {
    if (millis() - stateStartTime >= holdAtTopMs) {
      closeGate();
    }
  }

  delay(10); // small idle delay
}