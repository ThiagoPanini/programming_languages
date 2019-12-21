/*********************************************************
	     	     Semana 4 - Programa 15
	       Atividade 4.08 - Length statement
**********************************************************
1) Adicionar um LENGTH statement para prevenir dados truncados

***********************************************************/

data storm_summary2;
	set pg1.storm_summary;
	*Add a LENGTH statement;
	else Ocean="Pacific";
	keep Basin Season Name MaxWindMPH Ocean;
	*Add assignment statement;
	Basin=upcase(Basin);
	OceanCode=substr(Basin,2,1);
	if OceanCode="I" then Ocean="Indian";
	else if OceanCode="A" then Ocean="Atlantic";
	length Ocean $ 8;
run;