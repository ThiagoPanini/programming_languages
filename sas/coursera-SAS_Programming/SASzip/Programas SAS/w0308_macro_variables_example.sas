/*********************************************************
	     	     Semana 3 - Programa 8
	  	   Exemplo com Macro Variables: %LET
**********************************************************
	Em SAS, existe a possibilidade de criar variáveis para
evitar repetições desnecessárias de código. A criação dessas
variáveis são feitas a partir do comando %LET

***********************************************************/

/* Exemplo de utilização */
%let CarType="Wagon"; *Criação da variável;

proc print data=sashelp.cars;
	where Type=&CarType; *Utilizando variável com referência &;
	var Type Make Model MSRP;
run;

proc means data=sashelp.cars;
	where Type=&CarType;
	var MSRP MPG_Highway;
run;

proc freq data=sashelp.cars;
	where Type=&CarType;
	tables Origin Make;
run;