/* @file HelloKeypad.pde
|| @version 1.0
|| @author Alexander Brevig
|| @contact alexanderbrevig@gmail.com
||
|| @description
|| | Demonstrates the simplest use of the matrix Keypad library.
|| #
*/
#include <Keypad.h>

const byte ROWS = 2; //duas linhas
const byte COLS = 2; //duas colunas
char keys[ROWS][COLS] = {
  {'1','2'},
  {'3','4'}
};
byte rowPins[ROWS] = {5, 4}; //pinos do arduino que receberam os switchs "linhas"
byte colPins[COLS] = {8, 7}; //pinos do arduino que receberam os switchs "colunas"

Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );

void setup(){
  Serial.begin(9600);
}
  
void loop(){
  char key = keypad.getKey();
  
  if (key){
    Serial.println(key);
  }
}
