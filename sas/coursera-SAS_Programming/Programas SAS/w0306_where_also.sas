/*********************************************************
	     	     Semana 3 - Programa 6
	  		Atividade 3.02 - Testes com WHERE
**********************************************************
1) Execute o programa e examine os resultados no log.
	a) Os dois filtros WHERE foram aplicados?
2) Modifique o segundo WHERE e altere o código para WHERE ALSO.
Execute o código e examine os resultados no log.
	a) Os dois filtros WHERE foram aplicados?

***********************************************************/

proc print data=pg1.storm_summary;
	where MaxWindMPH>156;
	where MinPressure>800 and MinPressure<920;
run;

/* 1a) Não. Apenas o segundo filtro WHERE foi aplicado */

proc print data=pg1.storm_summary;
	where MaxWindMPH>156;
	where also MinPressure>800 and MinPressure<920;
run;

/* 2a) Sim. Ambos os filtro foram aplicados */