/*********************************************************
	     	     Semana 5 - Programa 6
	  Atividade 5.03 - Criando labels no data step
**********************************************************
1) Modificar o comando LABEL no DATA step para etiquetar
a coluna Invoice como sendo Invoice Price

2) Execute o programa. Por que os labels aparecem no
PROC MEANS mas não no PROC PRINT? Conserte o programa e
execute novamente

***********************************************************/

data cars_update;
    set sashelp.cars;
	keep Make Model MSRP Invoice AvgMPG;
	AvgMPG=mean(MPG_Highway, MPG_City);
	label MSRP="Manufacturer Suggested Retail Price"
          AvgMPG="Average Miles per Gallon"
          Invoice="Invoice Price";
run;

proc means data=cars_update min mean max;
    var MSRP Invoice;
run;

proc print data=cars_update label;
    var Make Model MSRP Invoice AvgMPG;
run;