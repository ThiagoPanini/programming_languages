/*********************************************************
	     	     Semana 6 - Programa 2
	Demonstração - Salvando arquivos através de libs
	
**********************************************************
	Nessa abordagem:
	- É criada uma biblioteca com o respectivo "engine"
	- Essa biblioteca utiliza uma referência de um nome de
arquivo (já existente ou não)
	- Ao criar os reports utilizando linguagem SAS, as tabelas
de output geradas são referenciadas (salvas) utilizando a lib
criada

***********************************************************
	     	     Primeira Parte - Exemplo 
	     	Exportando dados em um arquivo Excel
**********************************************************
	Sintaxe:
		LIBNAME libref XLSX "path/file.xlsx"
			<use libref for output table(s)>
		LIBNAME libref CLEAR;

***********************************************************/

%let outpath=~/EPG194/output;
libname myxlsx xlsx "&outpath/cars.xlsx";
data myxlsx.AsianCars;
	set sashelp.cars;
	where origin="Asia";
run;
libname myxlsx clear;

proc print data=myxlsx.AsianCars (obs=10); *Não funcionou. Seria o engine XLSX?;

/***********************************************************
	     	     Segunda Parte - Exemplo 
	     	Exportando dados em um arquivo Excel
**********************************************************
	Passo a passo:
1) Examinar as procedures DATA e MEANS para identificar as
tabelas temporárias que serão criadas. Executar o programa.

2) Adicionar um comando LIBNAME para criar uma lib chamada
xlout que aponta para um arquivo Excel chamado 
southpacific.xlsx na pasta "output"

3) Modificar os passos DATA e PROC para mandar o output das
tabelas referenciando a lib recém criada

4) Não esquecer de executar um CLEAR para a lib criada

***********************************************************/

libname xlout xlsx "&outpath/southpacific.xlsx";

data xlout.South_Pacific;
	set pg1.storm_final;
	where Basin="SP";
run;

proc means data=pg1.storm_final noprint maxdec=1;
	where Basin="SP";
	var MaxWindKM;
	class Season;
	ways 1;
	output out=xlout.Season_Stats n=Count mean=AvgMaxWindKM max=StrongestWindKM;
run;

libname xlout clear;