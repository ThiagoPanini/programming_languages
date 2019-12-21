/*********************************************************
	     	     Semana 4 - Programa 5
	  	Criando novas colunas em novas tabelas
**********************************************************
Exemplo de criação de novas colunas. Sintaxe:

	DATA output_table;
		SET input_table;
		new_column = expression;
	RUN;

***********************************************************/

data cars_new;
	set sashelp.cars;
	where Origin ne "USA";
	Profit = MSRP - Invoice; *nova coluna substraindo outras duas;
	Source = "Non-US Cars"; *nova coluna com string informativa;
	format Profit dollar10.;
	keep Make Model MSRP Invoice Profit Source;
run;

* Outro exemplo;
data tropical_storms;
	set pg1.storm_summary;
	drop Hem_EW Hem_NS Lat Lon;
	where Type='TS';
	*Criando nova coluna formatada;
	MaxWindKM=MaxWindMPH*1.60934;
	format MaxWindKM 3.; *arredonda para o número inteiro mais próximo;
	StormStype='Tropical Storm';
run;