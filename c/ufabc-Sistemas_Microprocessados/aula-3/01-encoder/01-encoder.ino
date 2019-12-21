/*AMS Rotary Sensor AS5040
 Measures absolute angle position referenced at a set NORTH

 Circuit
 UNO: MOSI pin 11
      MISO pin 12 
      CLK  pin 13
      CSN  pin 10

 Mega: MOSI pin 51
       MISO pin 50 
       CLK  pin 52
       CSN  pin 53  
 */

#include <stdio.h>

//Set Slave Select Pin
//MOSI, MISO, CLK are handeled automatically
  int CSN = 9;
  int D0 = 12;
  int CLK = 13 ;
  unsigned int angle = 0;
  int i;
  int test = 1;
  

void setup() {
  
  Serial.begin(9600);

  //Set Pin Modes
  pinMode(CSN, OUTPUT);
  pinMode(D0, INPUT);
  pinMode(CLK, OUTPUT);
  //Set Slave Select High to Start i.e disable chip
  digitalWrite(CSN, HIGH);

}

void loop() {

  digitalWrite(CSN, LOW);
  
  /*for(i=1; i<=10; i++) {
    digitalWrite(CLK, HIGH);
    //delayMicroseconds(1);
    digitalWrite(CLK, LOW); 
    angle = (angle << 1) | digitalRead(D0);
  }*/

  for(i=1; i<=10; i++) {
    digitalWrite(CLK, HIGH);
    delay(100);
    digitalWrite(CLK, LOW);
    delay(100); 

    angle = test << 1;
  }

  Serial.println(angle);
  
  //delay(250);
  
}
