/*********************************************************
	     	     Semana 4 - Programa 14
	   Atividade 4.07 - Trabalhando com condicionais
**********************************************************
1) Adicionar condicionais ELSE até uma condição verdadeira
ser encontrada.

2) Modificar o último IF-THEN para um ELSE

***********************************************************/

data storm_cat;
	set pg1.storm_summary;
	keep Name Basin MinPressure StartDate PressureGroup;
	*add ELSE keyword and remove final condition;
	if MinPressure=. then PressureGroup=.;
	else if MinPressure<=920 then PressureGroup=1;
	else PressureGroup=0;
run;

proc freq data=storm_cat;
	tables PressureGroup;
run;
