void setup() {
  pinMode(8, OUTPUT); // Choose any digital pin
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(8, HIGH);
    } else if (cmd == '0') {
      digitalWrite(8, LOW);
    }
  }
}
