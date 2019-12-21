/*********************************************************
	     	     Semana 5 - Programa 4
	  Exemplo - Segmentando reports através do BY
**********************************************************
Facilitando análises através de reports segmentados

***********************************************************/

proc sort data=sashelp.cars out=cars_sort;
	by Origin;
run;

ods noproctitle;
title "Origin Frequency Analysis";
proc freq data=cars_sort;
	by Origin;
	tables Type;
run;

ods proctitle;
title;