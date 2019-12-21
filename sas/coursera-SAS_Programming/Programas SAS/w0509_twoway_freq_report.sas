/*********************************************************
	     	     Semana 5 - Programa 9
	 		Exemplo - Two-ways frequency report
**********************************************************
	Adicionar um asterisco (*) entre duas colunas na
procedure FREQ cria um two-ways frequency report, ou seja,
um report de frequencia baseado no cruzamento entre dois
atributos. Vejamos abaixo um exemplo

	Sinxaxe:
	PROC FREQ DATA=input_table;
		TABLES col_name1*col_name2 </ options>;
	RUN;
	
	Parâmetros de customização
	PROC FREQ statement options:
		NO PRINT
	TABLES statement options:
		NOROW, NOCOL, NOPERCENT
		CROSSLIST, LIST
		OUT=output_table

***********************************************************/

* Análise de dados envolvendo a combinação entre Blood Pressure e Colesterol;
proc freq data=sashelp.heart;
	tables BP_Status*Chol_Status;
run;

/* Cada linha do report, para cada grupo, representa, na sequência, o que está disposto 
na tabela superior esquerda do report */

/*********************************************************
	     	     Outro exemplo de relatório
********************************************************* */
proc freq data=pg1.storm_final;
	tables BasinName*StartDate;
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;

/*********************************************************
	     	     Customizando relatório
********************************************************* */
proc freq data=pg1.storm_final;
	tables BasinName*StartDate / norow nocol nopercent;
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;

/*********************************************************
	     	Abordagem diferente utilizando CROSSLIST
********************************************************* */
proc freq data=pg1.storm_final;
	tables BasinName*StartDate / crosslist;
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;

/*********************************************************
	     	Abordagem diferente utilizando LIST
********************************************************* */
proc freq data=pg1.storm_final;
	tables BasinName*StartDate / list;
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;

/*********************************************************
	    Salvando resultado da procedure em tabela
********************************************************* */
proc freq data=pg1.storm_final noprint;
	tables BasinName*StartDate / out=stormcounts;
	format StartDate monname.;
	label BasinName="Basin"
		  StartDate="Storm Month";
run;