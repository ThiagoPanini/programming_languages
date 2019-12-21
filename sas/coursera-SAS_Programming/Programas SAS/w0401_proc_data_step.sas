/*********************************************************
	     	     Semana 4 - Programa 1
	  	       Demonstração - DATA step
**********************************************************
Entendendo o processo de subset com proc data.
    
***********************************************************/

/* Lendo dados e armazenando em outra tabela */
data myclass; *tabela my_class criada na lib WORK;
	set sashelp.class;
	where Age>=15;
	keep Name Age Height Weight; *colunas a serem transferidas pra my_class;
	format Height 4.1 Weight 3.1;
	*drop Sex; *alternativa=dropar colunas não desejadas em my_class;
run;

/* Comparando resultados */ 
title "Tabela sashelp.class original";
proc print data=sashelp.class (obs=5);
run;

title 'Tabela myclass da lib work (filtrada)';
proc print data=myclass (obs=5);
run;