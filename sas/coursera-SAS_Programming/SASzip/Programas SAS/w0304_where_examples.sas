/*********************************************************
	     	     Semana 3 - Programa 4
	  		Filtrando consulta com WHERE
**********************************************************
Sintaxe:

proc <procedure-name>...;
	where <expression>;
run;

Onde <expression> = column operator value

***********************************************************/
/* Exemplos de utilização do WHERE
   1) operador AND */
proc print data=sashelp.cars;
	var Make Model Type MSRP MPG_City MPG_Highway;
	where Type="SUV" and MSRP <= 30000;
run;

/* 2) operador OR */
proc print data=sashelp.cars;
	var Make Model Type MSRP MPG_City MPG_Highway;
	where Type="SUV" or Type="Truck" or Type="Wagon";
run;

/* 3) operadores IN e NOT IN */
proc print data=sashelp.cars;
	var Make Model Type MSRP MPG_City MPG_Highway;
	where Type IN ("SUV", "Truck", "Wagon")
run;
