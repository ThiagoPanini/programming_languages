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

int geladeira_potencia=300;
int tv_potencia=90;
int freezer_potencia=400;
int computador_potencia=300;

int geladeira_consumo=90;
int tv_consumo=15;
int freezer_consumo=120;
int computador_consumo=20;

int CONSUMO_TOTAL=0;

int inicio=0;
int apaga=0;

const byte ROWS = 1; 
const byte COLS = 4; 
char keys[ROWS][COLS] = {
        {'1','2','3','4'}
        };
byte rowPins[ROWS] = {8}; //vermelho
byte colPins[COLS] = {7, 6, 5, 4}; //laranja, azul, amarelo e marrom

Keypad kpd = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );

uint8_t gHue = 0; 

void sinelon()
{  
  fadeToBlackBy( leds, NUM_LEDS, 20);
  int pos = beatsin16( 13, 0, NUM_LEDS-1 );
  leds[pos] += CHSV( gHue, 255, 192);
  if (pos==0) inicio=1; 
}

void setup() {
    //lcd.init();
    //lcd.backlight();
    FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);           
    startTime = millis();
    for (int i=0; i<NUM_LEDS; i++) meus_leds[i]=CRGB::Black;
}

void loop() {  
  if ( (millis()-startTime)>60000 ) {
    startTime = millis();  
    inicio=0;
  }  

/*  if ( (millis()-clearTime)>5000 ) {
    clearTime = millis();  
    lcd.setCursor(0,1);
    lcd.print("                    ");  
    lcd.setCursor(0,2);
    lcd.print("                    ");      
    lcd.setCursor(0,3);
    lcd.print("                    ");         
  }*/    
  
  if (inicio==0)
  {
    sinelon();
    FastLED.show();        
    FastLED.delay(1000/FRAMES_PER_SECOND);         
    EVERY_N_MILLISECONDS( 20 ) { gHue++; }      
    fadeToBlackBy( leds, NUM_LEDS, 20);
    leds[0] = CRGB::Black;
    leds[1] = CRGB::Black;
    leds[2] = CRGB::Black; 
    leds[3] = CRGB::Black; 
    leds[4] = CRGB::Black; 
    FastLED.show();               
  }
  else
  {                 
    if (kpd.getKeys())
    {
        for (int i=0; i<LIST_MAX; i++)   
        {
            if ( kpd.key[i].stateChanged )   
            {
                startTime = millis();
                clearTime = millis();
                
                switch (kpd.key[i].kstate) {  
                case HOLD: 
                    lcd.clear();                
                    switch(kpd.key[i].kchar){
                      case  '1':
                        CONSUMO_TOTAL+=geladeira_consumo;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.println(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Geladeira");
                        Serial.print(": ");
                        Serial.println(geladeira_potencia);
                        Serial.print("Watts");                             
                        if (geladeira_consumo<MED_POT) meus_leds[0] = CRGB::Blue;
                        else meus_leds[0] = CRGB::Red;                        
                        break;
                      case  '2':
                        CONSUMO_TOTAL+=tv_consumo;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.print(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Televisor");
                        Serial.print(": ");
                        Serial.println(tv_potencia);
                        Serial.print("Watts");                                            
                        if (tv_consumo<MED_POT) meus_leds[1] = CRGB::Blue;
                        else meus_leds[1] = CRGB::Red;                        
                        break;  
                      case  '3':
                        CONSUMO_TOTAL+=freezer_consumo;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.println(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Freezer");
                        Serial.print(": ");
                        Serial.print(freezer_potencia);
                        Serial.println("Watts");                      
                        if (freezer_consumo<MED_POT) meus_leds[2] = CRGB::Blue;
                        else meus_leds[2] = CRGB::Red;                        
                        break; 
                      case  '4':
                        CONSUMO_TOTAL+=computador_consumo;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.println(CONSUMO_TOTAL);
                        Serial.print("Kwh");
                        Serial.print("Computador");
                        Serial.print(": ");
                        Serial.println(computador_potencia);
                        Serial.print("Watts");              
                        if (computador_consumo<MED_POT) meus_leds[3] = CRGB::Blue;
                        else meus_leds[3] = CRGB::Red;                        
                        break;                                                                   
                    }                                      
                break;                
                case RELEASED:                                  
                    switch(kpd.key[i].kchar){
                      case  '1':
                        CONSUMO_TOTAL-=geladeira_consumo;
                        if (CONSUMO_TOTAL<0) CONSUMO_TOTAL=0;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.println(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Geladeira");                       
                        Serial.print(" Desligada");
                        meus_leds[0] = CRGB::Black;                        
                        break;
                      case  '2':
                        CONSUMO_TOTAL-=tv_consumo;
                        if (CONSUMO_TOTAL<0) CONSUMO_TOTAL=0;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.print(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Televisor");                       
                        Serial.print(" Desligado");
                        meus_leds[1] = CRGB::Black;                        
                        break; 
                      case  '3':
                        CONSUMO_TOTAL-=freezer_consumo;
                        if (CONSUMO_TOTAL<0) CONSUMO_TOTAL=0;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.print(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Freezer");                       
                        Serial.print(" Desligado");
                        meus_leds[2] = CRGB::Black;                        
                        break;
                      case  '4':
                        CONSUMO_TOTAL-=computador_consumo;
                        if (CONSUMO_TOTAL<0) CONSUMO_TOTAL=0;
                        Serial.print("CONSUMO TOTAL:");
                        Serial.print(CONSUMO_TOTAL);
                        Serial.println("Kwh");
                        Serial.print("Computador");                       
                        Serial.print(" Desligado");
                        meus_leds[3] = CRGB::Black;                        
                        break;                                            
                    }                     
                break;                    
                }
            for (int i=0; i<NUM_LEDS; i++) leds[i]=meus_leds[i];
            FastLED.show();                         
            }
        }
    }    
  }  
}  
