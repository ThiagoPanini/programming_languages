/*********************************************************
	     	     Semana 3 - Programa 13
	       Atividade 3.07 - Ordenando dados
**********************************************************
Sintaxe:

	PROC SORT data=input_table <OUT=output_table>;
		BY <DESCENDING> col_name(s);
	run;
	
O comando opcional OUT cria uma tabela temporária com os
dados ordenados.

***********************************************************/

proc sort data=pg1.storm_summary out=storm_sort;
	by descending MaxWindMPH;
	where Basin in ("NA", 'na');
run;
