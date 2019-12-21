/*********************************************************
	     	     Semana 4 - Programa 19
	Prática Level 2 - Condicional com comando DO
**********************************************************
1) Dividir a tabela pg1.np_summary em duas tabelas:
	parks e monuments

2) Filtrar a tabela apenas por Parques Nacionais
(Type=NP ou NM) e, após isso, aplicar a condicional

3) Criar uma coluna chamada Campers para alocar a soma de todas
as colunas que contém contagem de campers. Formatar
coluna para incluir vírgular

4) Quando o tipo for NP, criar uma nova coluna chamada 
ParkType que recebe "Park" e adicionar uma linha na tabela
parks. Quando o tipo for NM, a variável ParkType recebe
"Monument" e este valor é adicionado a tabela monuments

5) Manter Reg, ParkName, DayVisits, OtherLodging, Campers e 
ParkType de ambas as tabelas

	a) Qual tabela tem a maior quantidade de linhas?

***********************************************************/

proc print data=pg1.np_summary (obs=5);
run;

proc freq data=pg1.np_summary;
	tables Type;
run;

/* Lendo dados */
data parks monuments;
	set pg1.np_summary;
	keep Reg ParkName DayVisits OtherLodging Campers Type ParkType;
	where Type in ("NM", "NP");
	Campers=SUM(OtherCamping, TentCampers, RVCampers, BackcountryCampers);
	format Campers comma7.;
	length ParkType $ 8;
	if Type='NP' then do;
		ParkType='Park';
		output parks;
	end;
	else if Type='NM' then do;
		ParkType='Monument';
		output monuments;
	end;
run;

proc print data=parks;

proc print data=monuments;