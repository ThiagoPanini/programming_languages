/*
* Ultrasonic Sensor HC-SR04 and Arduino Tutorial
*
* by Dejan Nedelkovski,
* www.HowToMechatronics.com
*
*/
#include <Ultrasonic.h>
#include <Keypad.h>
#include <FastLED.h>
#include <Wire.h> 

#define NUM_LEDS 21
#define DATA_PIN 2   // marrom (vindo dos leds)
#define BRIGHTNESS          96
#define FRAMES_PER_SECOND  120
#define MED_POT 100

CRGB leds[NUM_LEDS];
CRGB meus_leds[NUM_LEDS];

unsigned long startTime;
unsigned long clearTime;

// defines pins numbers
const int trigPin = 9;
const int echoPin = 10;

// defines variables
long duration;
int distance;
int inicio=0;
int apaga=0;

uint8_t gHue = 0; 

//Função sinelon
void sinelon()
{  
  fadeToBlackBy( leds, NUM_LEDS, 20);
  int pos = beatsin16( 13, 0, NUM_LEDS-1 );
  leds[pos] += CHSV( gHue, 255, 192);
  if (pos==0) inicio=1; 
}

void setup() {
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
  Serial.begin(9600); // Starts the serial communication
  FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);        
  startTime = millis();   
  for (int i=0; i<NUM_LEDS; i++) meus_leds[i]=CRGB::Black;
  
}

void loop() {
  inicio=0;
  
  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delay(100);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delay(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance= duration*0.034/2;
  // Prints the distance on the Serial Monitor
  Serial.print("Distance: ");
  Serial.println(distance);

  //Tratando LEDs
  if (inicio==0)
  {
    sinelon();
    FastLED.show();        
    FastLED.delay(1000/FRAMES_PER_SECOND);         
    EVERY_N_MILLISECONDS( 20 ) { gHue++; }      
    fadeToBlackBy( leds, NUM_LEDS, 20);
    for (int i=0; i<NUM_LEDS; i++) meus_leds[i]=CRGB::Black; 
    FastLED.show();  
    inicio=1;            
  }

  for (int i=0; i<NUM_LEDS; i++) meus_leds[i]=CRGB::Blue;
  
  
  for(int dot = 0; dot < NUM_LEDS; dot++) { 
      leds[dot] = CRGB::Blue;
      FastLED.show();
      // clear this led for the next time around the loop
      leds[dot] = CRGB::Black;
      delay(30);
  }
  FastLED.setBrightness(distance*10); 

  for (int i=0; i<NUM_LEDS; i++) leds[i]=meus_leds[i];
            FastLED.show();    
            
}
