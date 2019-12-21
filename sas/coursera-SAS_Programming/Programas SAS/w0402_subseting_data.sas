/*********************************************************
	     	     Semana 4 - Programa 2
	  	Atividade 4.03 - utilizando o proc data
**********************************************************
1) Implementar o data step para ler a tabela pg1.storm_summary
e criar uma tabela de output chamada Storm_cat5 como uma
tabela permanente no caminho EPG194/output

2) Incluir apenas as tempestades de Categoria 5: 
MaxWindMPH maior ou igual 156 com StartDate após 01JAN2000

3) Incluir um passo para adicionar as seguintes colunas na
tabela de output: Season, Basin, Name, Type e MaxWindMPH
	a) Quantas tempestades de Categoria 5 contidas?

***********************************************************/
libname out "~/EPG194/output";
%let max_wind=156;
%let start_date=01jan2000; 

data out.Storm_cat5;
	set pg1.storm_summary;
	where MaxWindMPH ge &max_wind and StartDate >= "&start_date"d;
	keep Season Basin Name Type MaxWindMPH;
run;
	