/*********************************************************
	     	     Semana 5 - Programa 11
			Prática Level 2 - Proc FREQ
**********************************************************
	A tabela pg1.np_codelookup é usada para procurar nomes
de parques ou seus respectivos códigos. Contudo, essa tabela
também inclui colunas para o tipo do parque e região.

1) Crie um passo PROC FREQ para analisar as linhas da tabela
pg1.np_codelookup

2) Gerar um two-way frequency report para Type e Region

3) Excluir parques que contém a palavra "Order" em Type

4) Ordenar o resultado por frequência no modo descendente

5) Não mostrar as porcentagens no report

6) Inserir o título "Park Types by Region"
	
	a) Quais são os top 3 parques baseados na frequência

***********************************************************/

* Parte 1 - Criando um report;
proc print data=pg1.np_codelookup (obs=5);
run;

ods noproctitle;
title "Park Types by Region";
proc freq data=pg1.np_codelookup order=freq;
	tables Type*Region / nopercent;
	where Type not like "%Other%";
run;
ods proctitle;
title;

* a) National Historic Site, National Monument e National Park;

/**********************************************************
	Modificar o programa para incluir as seguintes alterações:

1) Limitar os tipos de parques para os três mais frequentes
(resposta da questão anterior)

2) Em adição a eliminação das colunas de porcentagem, adicionar
a opção CROSSLIST no report

3) Adicionar um plot de frequência para agrupamento dos dados
em linhas, mostrando as linhas em porcentagem e com orientação 
horizontal.
Nota: utilizar a documentação para aprender GROUPBY=, SCALE=
e ORIENT= 

4) Título "Selected Park Types by Region"

	a) Qual região tem o maior valor de porcentagem em linha?

***********************************************************/

ods graphics on;
ods noproctitle;
title "Selected Park Types by Region";
proc freq data=pg1.np_codelookup order=freq;
	tables Type*Region / nocol crosslist
						plots=freqplot(groupby=row orient=horizontal scale=percent);
	where Type in ('National Historic Site', 
						'National Monument',
						'National Park');
run;
ods proctitle;
title;

/***********************************************************
	Personalização de acordo com programa p105p03.sas
***********************************************************/

title1 'Counts of Selected Park Types by Park Region';
ods graphics on;
proc freq data=pg1.np_codelookup order=freq;
	tables Type*Region / crosslist plots=freqplot(twoway=stacked orient=horizontal);
	where type in ('National Historic Site', 'National Monument', 'National Park');
run;

/*part b*/
title1 'Counts of Selected Park Types by Park Region';
proc sgplot data=pg1.np_codelookup;
    where Type in ('National Historic Site', 'National Monument', 'National Park');
    hbar region / group=type;
    keylegend / opaque across=1 position=bottomright location=inside;
    xaxis grid;
run;
