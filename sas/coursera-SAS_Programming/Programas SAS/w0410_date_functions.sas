/*********************************************************
	     	     Semana 4 - Programa 10
	           Exemplo - Funções de Data
**********************************************************
Criar duas colunas:
	a) YearsPassed: quantos anos se passaram desde a data
informada na tabela até a data atual
	b) Anniversary: aniversário da tempestade no ano atual
***********************************************************/

/* Criando tabela temporária e armazenando dados */
data storm_new;
	set pg1.storm_damage;
	drop Summary;
	YearsPassed = YRDIF(Date, TODAY(), "age");
	Anniversary = MDY(MONTH(Date), DAY(Date), YEAR(TODAY()));
	format YearsPassed 4.1 Date Anniversary mmddyy10.;
run;