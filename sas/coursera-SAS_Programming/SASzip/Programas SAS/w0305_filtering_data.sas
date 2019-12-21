/*********************************************************
	     	     Semana 3 - Programa 5
	  		Filtros básicos em tabela (WHERE)
**********************************************************
	O objetivo desse programa é a familiarização com as
cláusulas where para filtragem de dados em uma tabela

***********************************************************/

/* 1) MaxWindMPH maior ou igual a 156 */
proc print data=pg1.storm_summary;
	where MaxWindMPH >= 156;
run;

/* 2) Basin igual a "WP" */
proc print data=pg1.storm_summary;
	where Basin = 'WP'; * Case Sensitive;
run;

/* 3) Valores para mais de um Basin */
proc print data=pg1.storm_summary;
	where Basin IN ('SI', 'NI'); * Case Sensitive;
run;

/* 4) Filtrando datas (SAS dates constant d) */
proc print data=pg1.storm_summary;
	where StartDate >= "01jan2010"d; * SAS date constant;
run;

/* 5) Type="TS" and Hem_EW="W" */
proc print data=pg1.storm_summary;
	where Type="TS" and Hem_EW="W";
run;

/* 6) MaxWindMPH > 156 ou MinPressure<920 */
proc print data=pg1.storm_summary;
	where MaxWindMPH>156 or MinPressure<920;
	/* MinPressure tem missing (menor valor possível) */
run;

/* 7) Tratando missing data em MinPressure */
proc print data=pg1.storm_summary;
	where MaxWindMPH>156 or 0<MinPressure<920;
	/* MinPressure tem missing (menor valor possível) */
run;