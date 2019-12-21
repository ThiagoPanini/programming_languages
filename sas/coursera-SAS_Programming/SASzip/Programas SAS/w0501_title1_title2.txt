/*********************************************************
	     	     Semana 5 - Programa 1
				Atividade 5.01 - TITLE
**********************************************************
1) Adicionar títulos nas procs

***********************************************************/

title "Storm Analysis";
title2 'Summary Statistics for MaxWind and MinPressure';
proc means data=pg1.storm_final;
	var MaxWindMPH MinPressure;
run;

title2 'Frequency Report for Basin';
proc freq data=pg1.storm_final;
	tables BasinName;
run;