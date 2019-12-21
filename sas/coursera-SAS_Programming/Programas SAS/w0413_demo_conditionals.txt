/*********************************************************
	     	     Semana 4 - Programa 13
	     Demonstração - Condicionais IF THEN
**********************************************************
Criando um novo grupo baseado em uma condição específica

***********************************************************/

*Lendo dados da tabela sashelp.cars;
data cars2;
	set sashelp.cars;
	if MSRP<30000 then Cost_Group=1;
	if MSRP>=30000 then Cost_Group=2;
	keep Make MOdel Type MSRP Cost_Group;
run;

*Lendo dados da tabela storm_summary;
data storm_new;
	set pg1.storm_summary;
	keep Season Name Basin MinPressure PressureGroup;
	* Cuidando dos dados missing;
	if MinPressure=. then PresureGroup=.;
	else if MinPressure<=920 then PressureGroup=1; *Sem o else, essa condição sobescreve a primeira (Null=menor valor possível);
	else if MinPressure>920 then PressureGroup=0;
run;