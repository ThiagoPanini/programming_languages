# define cs 19
# define d0 12
# define clk 13

void setup() {

  pinMode(d0, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

}

void loop() {

  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  
}
