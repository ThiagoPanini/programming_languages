/*********************************************************
	     	     Semana 4 - Programa 16
	   Exemplo - Gerando duas tabelas com IF THEN DO
**********************************************************
Utilizado do comando IF THEN DO

***********************************************************/

data under40 over40;
	set sashelp.cars;
	keep Make MOdel msrp cost_group;
	if msrp<20000 then do;
		cost_group=1;
		output under40;
	end;
	else if mrsp<40000 then do;
		cost_group=2;
		output over40;
	end;
	else do;
		cost_group=3;
		output over40;
	end;
run;