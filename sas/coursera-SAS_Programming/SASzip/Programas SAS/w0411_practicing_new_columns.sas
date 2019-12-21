/*********************************************************
	     	     Semana 4 - Programa 11
	     Prática Level 1 - Novas Colunas com Funções
**********************************************************
1) Criar uma nova tabela chamada np_summary_update a partir
dos dados da tabela pg1.np_summary. Criar três novas colunas:
SqMiles, Camping e TypeName.

2) A coluna SqMiles será dada pela multiplicação da coluna
Acres por 0.0015625;

3) A coluna Camping receberá a soma de OtherCamping, 
TentCampers, RVCampers e BackcountryCampers

4) Formatar as colunas SqMiles e Camping para incluir vírgulas
e nenhuma casa decimal;

5) Criar um comando KEEP para incluir as novas colunas

	a) Qual o valor de Camping na primeira linha?
	b) Qual o valor de SqMiles na primeira linha?
***********************************************************/

* Conhecendo os dados da tabela original;
proc print data=pg1.np_summary(obs=10);
run;

* Modificando dados;
data np_summary_update;
	set pg1.np_summary;
	SqMiles = Acres*0.0015625;
	Camping = SUM(OtherCamping, TentCampers, RVCampers, BackcountryCampers);
	keep Reg ParkName DayVisits OtherLodging Acres SqMiles Camping;
	format SqMiles COMMA6. Camping COMMA9.
run;

/* a) SqMiles = 1,014
   b) Camping = 6,375 */