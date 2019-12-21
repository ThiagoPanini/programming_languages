/*
   Curso de Arduino e AVR WR Kits Channel
   
   Aula 88 - RFID - Introdução
   
    
   Autor: Eng. Wagner Rambo  Data: Outubro de 2016
   
   www.wrkits.com.br | facebook.com/wrkits | youtube.com/user/canalwrkits
   
*/

// --- Bibliotecas Auxiliares ---
#include <SPI.h>
#include <MFRC522.h>


// --- Mapeamento de Hardware ---
#define SS_PIN 10
#define RST_PIN 9
MFRC522 mfrc522(SS_PIN, RST_PIN);   // Cria instância com MFRC522
 

// --- Variáveis Globais --- 
char st[20];
boolean libera = false;
int LED_libera = 5;


// --- Configurações Iniciais ---
void setup() 
{

  Serial.begin(9600);   // Inicia comunicação Serial em 9600 baud rate
  SPI.begin();          // Inicia comunicação SPI bus
  mfrc522.PCD_Init();   // Inicia MFRC522
  
  //Serial.println("Aproxime o seu cartao do leitor...");
  //Serial.println();
   
  
} //end setup


// --- Loop Infinito ---
void loop() 
{
  // Verifica novos cartões
  if ( ! mfrc522.PICC_IsNewCardPresent()) 
  {
    return;
  }
  // Seleciona um dos cartões
  if ( ! mfrc522.PICC_ReadCardSerial()) 
  {
    return;
  }
  
  // Mostra UID na serial
  //Serial.print("UID da tag :");
  //Serial.println();
  String conteudo= "";
  byte letra;
  for (byte i = 0; i < mfrc522.uid.size; i++) 
  {
     Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
     Serial.print(mfrc522.uid.uidByte[i], HEX);
     conteudo.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     conteudo.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  //Serial.println();
  //Serial.print("Mensagem : ");
  conteudo.toUpperCase();
  //Serial.print(conteudo);
  Serial.println();
  delay(1000);
} 
/*  if (conteudo.substring(1) == "A1 FA B4 63") //UID 1 - Chaveiro
  {
    Serial.println("Chaveiro do kit identificado!");
    Serial.println();
    libera = true;
    delay(3000);
 
  }
 
  if (conteudo.substring(1) == "CD C7 35 7D") //Cartao Lucas Vellozo
  {
    Serial.println("Lucas Vellozo de Sousa, seja bem-vindo à UFABC!");
    Serial.println("Passagem liberada.");
    Serial.println();
    libera = true;
    delay(3000); 
  }

  if (conteudo.substring(1) == "84 FC 45 72") //UID 2 - Cartao
  {
    Serial.println("Thiago Panini, seja bem-vindo à UFABC!"); //Cartão Thiago Panini
    Serial.println("Passagem liberada.");
    Serial.println();
    libera = true;
    delay(3000); 
  }

  // Aciona saída/entrada
  if (libera)
  {
    digitalWrite(4, HIGH);
    delay(5000);
    digitalWrite(4, LOW);
  }
  
  delay(50);
  
  
} //end loop
 */
 
