/*********************************************************
	     	     Semana 4 - Programa 9
	          Atividade 4.06 - Substr Function
**********************************************************
1) Adicionar um filtro WHERE utilizando a função SUBSTR
para retornar apenas a segunda letra da coluna Basin da
tabela pg1.storm_summary (armazenar resultado em outra
tabela)

	a) Quantas tempestades estão na Pacific Basin (P)

***********************************************************/

proc print data=pg1.storm_summary(obs=5);

data ocean_storm;
	set pg1.storm_summary;
	OceanCode=substr(Basin, 2, 1);
	where substr(Basin, 2, 1)="P";
run;

proc print data=ocean_storm (obs=5);