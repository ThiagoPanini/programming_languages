/*********************************************************
	     	     Semana 2 - Programa 3
	   Prática 1 - Importando arquivo em excel
				   Salvando tabela em uma biblioteca
**********************************************************
   Conteúdo do programa:

1) Importando arquivo em excel (proc import)
   a) opções: datafile, dbms=xlsx, out=tabela, replace
2) Questões:
   a) Quantas variáves existem na tabela eu_sport_trade?
 
***********************************************************/

proc import datafile="~/EPG194/data/eu_sport_trade.xlsx"
			dbms=xlsx out=eu_sport_trade replace;
run;

proc contents data=eu_sport_trade;
run;

