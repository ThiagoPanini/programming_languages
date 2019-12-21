/*********************************************************
	     	     Semana 3 - Programa 10
	  	    Atividade 3.04 - Variáveis com %LET
**********************************************************
1) Modificar o valor da variável Basin de "NA" para "SP"

2) Executar o programa e verificar o log.
	a) Qual procedure não apresentou um report?
	b) Qual a diferença no WHERE dessa procedure?

***********************************************************/

%let BasinCode=SP;

proc means data=pg1.storm_summary;
	where Basin="&BasinCode";
	var MaxWindMPH MinPressure;
run;

proc freq data=pg1.storm_summary;
	where Basin='&BasinCode';
	tables Type;
run;

/* 
	a) PROC FREQ não produziu os reports
	b) A diferença são as aspas duplas no primeiro caso ("") e as aspas simples no segundo ('')
*/