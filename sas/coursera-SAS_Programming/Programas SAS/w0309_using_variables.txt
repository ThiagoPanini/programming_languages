/*********************************************************
	     	     Semana 3 - Programa 9
	  	    Utilizando variáveis na prática
**********************************************************
	Sintaxe de declaração:
		%LET <var_name> = <value>;
	
	Sintaxe de chamada:
		WHERE numvar=&macrovar;
		WHERE charvar="&macrovar";
		WHERE datevar="&macrovar"d;

***********************************************************/

proc print data=pg1.storm_summary;
	where MaxWindMPH>=156 and Basin="NA" and StartDate>="01jan2000"d;
	var Basin Name StartDate EndDate MaxWindMPH;
run;

proc means data=pg1.storm_summary;
	where MaxWindMPH>=156 and Basin="NA" and StartDate>="01jan2000"d;
	var MaxWindMPH MinPressure;
run;

/* Criando variáveis e refazendo o código */
%let max_wind_mph=156;
%let basin="NA";
%let start_date="01jan2000"d;

proc print data=pg1.storm_summary;
	where MaxWindMPH>=&max_wind_mph and Basin=&basin and StartDate>=&start_date;
	var Basin Name StartDate EndDate MaxWindMPH;
run;

proc means data=pg1.storm_summary;
	where MaxWindMPH>=156 and Basin="NA" and StartDate>="01jan2000"d;
	var MaxWindMPH MinPressure;
run;

