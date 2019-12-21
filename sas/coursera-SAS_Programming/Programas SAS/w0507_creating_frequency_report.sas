/*********************************************************
	     	     Semana 5 - Programa 7
	     Demo - Criando um report de frequências
**********************************************************
	Através de algumas opções passadas como parâmetros para
a procedure FREQ, é possível customizar um report de 
frequências de diversas formas.

1) Executar a primeira parte do programa para visualizar o
report sem nenhuma configuração adicional

2) Executar a segunda parte do programa já com os parâmetros
modificados e com o report customizado com as opções

	order=report_col (ordena tabela de acordo com variável do report)
	nlevels (cria uma tabela inicial com a quantidade de entradas únicas p/ cada coluna)
	/ nocum (elimina colunas cumulativas)
	format (pode ser utilizado para agrupar dados, como datas, por exemplo)
	ods graphics on (permite a criação de gráficos no report)
	/ plots=plot_type(<options>) (cria e customiza gráficos)

***********************************************************/

/* 1) Report de Frequências sem customização */
proc freq data=pg1.storm_final;
	tables BasinName Season;
run;

/* 2) Report de Frequências customizado */
ods graphics on;
ods noproctitle;
title "Frequency Report for Basin and Storm Month";
proc freq data=pg1.storm_final order=freq nlevels;
	tables BasinName StartDate / nocum plots=freqplot(orient=horizontal scale=percent);
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;
title;
ods proctitle;