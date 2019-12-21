/*********************************************************
	     	     Semana 3 - Programa 3
	  Prática 2 - Explorando dados com comando proc
**********************************************************
Objetivos:

A tabela pg1.np_summary contém dados sobre parques nacionais
dos EUA, monumentos, preservação, rios e praias.
 
***********************************************************/

/* 1) Executar PROC FREQ em uma nova janela para as colunas
Reg e Type. Existem valores inválidos para a coluna Reg? 
	2) Existem valores inválidos para a coluna Type?*/
proc freq data=pg1.np_summary;
	tables Reg Type;
run;

/* 3) Executar PROC UNIVARIATE para a coluna Acres. Qual o nº
de observação para o menor parque? */
proc univariate data=pg1.np_summary;
	var Acres;
run;

/* 4) Encontre a observação para o maior parque nos resultados
do comando PROC UNIVARIATE. Visualize a linha na tabela
Qual é o nome do maior parque? */
proc print data=pg1.np_summary;
	var ParkName Acres;
run;