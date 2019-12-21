# define cs 9
# define d0 12
# define clk 13

int counter = 0; 
int ultimoEstado;
int atualEstado;

void setup() {  
  pinMode(clk, OUTPUT);
  pinMode(cs, OUTPUT);
  pinMode(d0, INPUT);

  Serial.begin (9600);
  ultimoEstado = digitalRead(clk);
}

void loop() {
  atualEstado = digitalRead(clk);
  if (ultimoEstado != atualEstado) {
    if (digitalRead(d0) != atualEstado) {
      counter++;
    } else  {
      counter--;
    }

    Serial.print("Position: ");
    Serial.println(counter);
  }
  ultimoEstado = atualEstado;

}
