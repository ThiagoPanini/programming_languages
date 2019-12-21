/*********************************************************
	     	     Semana 2 - Programa 5
	   Prática 2 - Importando arquivo csv
				   Salvando tabela em uma biblioteca
**********************************************************
   Conteúdo do programa:

1) Importando arquivo em excel (proc import)
   a) opções: datafile, dbms=csv, out=tabela, replace
2) Examinar os dados
3) Evitar dados truncados (guessingrows=max)
 
***********************************************************/

*Lendo os dados;
proc import datafile="~/EPG194/data/np_traffic.csv"
			dbms=csv out=np_traffic replace; 
guessingrows=max;
run;

*Examinando dataset;
proc contents data=np_traffic;
run;