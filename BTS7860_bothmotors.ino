// ===== BTS7960 TWO MOTOR CONTROL (WASD) =====

// LEFT MOTOR
int L_RPWM = 5;
int L_LPWM = 6;
int L_EN   = 2;
int L_LEN  = 3;

// RIGHT MOTOR
int R_RPWM = 9;
int R_LPWM = 10;
int R_EN   = 7;
int R_LEN  = 8;



char cmd;
int speed = 160; // 0–255

void setup() {
  pinMode(R_RPWM, OUTPUT);
  pinMode(R_LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(R_LEN, OUTPUT);

  pinMode(L_RPWM, OUTPUT);
  pinMode(L_LPWM, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(L_LEN, OUTPUT);

  // Enable both drivers
  digitalWrite(R_EN, HIGH);
  digitalWrite(R_LEN, HIGH);
  digitalWrite(L_EN, HIGH);
  digitalWrite(L_LEN, HIGH);

  Serial.begin(9600);
}

void stopMotors() {
  analogWrite(R_RPWM, 0);
  analogWrite(R_LPWM, 0);
  analogWrite(L_RPWM, 0);
  analogWrite(L_LPWM, 0);
}

// ===== FORWARD =====
// Motors are mirrored, so directions are opposite
void forward() {
  analogWrite(R_RPWM, speed);
  analogWrite(R_LPWM, 0);

  analogWrite(L_RPWM, 0);
  analogWrite(L_LPWM, speed);
}

// ===== BACKWARD =====
void backward() {
  analogWrite(R_RPWM, 0);
  analogWrite(R_LPWM, speed);

  analogWrite(L_RPWM, speed);
  analogWrite(L_LPWM, 0);
}

// ===== TURN LEFT =====
void left() {
  analogWrite(R_RPWM, speed);
  analogWrite(R_LPWM, 0);

  analogWrite(L_RPWM, 0);
  // analogWrite(L_LPWM, speed / 3);
  analogWrite(L_LPWM, 0);
}

// ===== TURN RIGHT =====
void right() {
  //analogWrite(R_RPWM, speed / 3);
  analogWrite(R_RPWM, 0);
  analogWrite(R_LPWM, 0);

  analogWrite(L_RPWM, 0);
  analogWrite(L_LPWM, speed);
}

void loop() {
  if (Serial.available()) {
    cmd = Serial.read();

    switch (cmd) {
      case 'F': forward(); break;  // W
      case 'B': backward(); break; // S
      case 'L': left(); break;     // A
      case 'R': right(); break;    // D
      case 'S': stopMotors(); break;
    }
  }
}
