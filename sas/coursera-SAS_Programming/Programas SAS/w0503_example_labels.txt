/*********************************************************
	     	     Semana 5 - Programa 3
	  Exemplo - Utilizando label como um allias
**********************************************************
Sintaxe:
	LABEL <column>="new_column_name"

***********************************************************/

proc means data=sashelp.cars;
	where type="Sedan";
	var MSRP MPG_Highway;
	label MSRP="Manufacter Suggested Retail Price"
		  MPG_Highway="Highway Miles per Gallon";
run;