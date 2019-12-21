/*********************************************************
	     	     Semana 3 - Programa 1
	  Demo 1 - Explorando dados com comando proc
**********************************************************
	Sintaxe das diferentes procs em SAS:
	
	* ---------------- PROC PRINT ------------------
	Lista linhas e colunas de uma tabela
	
	PROC PRINT DATA=input_table(OBS=n);
		VAR col_name(s);
	RUN;
	
	* ---------------- PROC MEANS ------------------
	Calcula estatísticas de colunas em uma tabela
	(n, mean, std_dev, min e max). Utilizada para:
		- Dados missing
		- Range para indicar valores indevidos
		
	PROC MEANS DATA=input_table;
		VAR col_name(s);
	RUN;
	
	* ---------------- PROC UNIVARIATE ------------------
	Também relacionada a estatistíca da tabela, porém de uma
	forma mais profunda.
	
	PROC UNIVARIATE DATA=input_table;
		VAR col_name(s);
	RUN;
	
	* ---------------- PROC FREQ ------------------
	Cria uma visão de frequências para cada coluna na tabela
	(freq, percent, freq acumulada e porcentagem acumulada)
	PROC FREQ DATA=input_table;
		TABLES col_name(s);
	RUN;
 
***********************************************************/

/* List 10 first rows */
proc print data=pg1.storm_summary (obs=10);
	* Limitando as colunas;
	var  Season Name MaxWindMPH MinPressure StartDate EndDate Basin;
run;

/* Calculate summary statistics */
proc means data=pg1.storm_summary;
	* No proc means, as colunas devem ser numéricas;
	var MaxWindMPH MinPressure;
run;

/* Examine extreme values */
proc univariate data=pg1.storm_summary;
	* No proc univariate, temos um "relatório" p/ c/ var;
	var MaxWindMPH MinPressure;
run;

/* Counting values from attributes */
proc freq data=pg1.storm_summary;
	* No proc freq, temos "tables" e não "var";
	tables Basin Type Season;
run;