/*********************************************************
	     	     Semana 6 - Programa 10
		Atividade 7.03 - CREATE TABLE SQL no SAS
	
**********************************************************
	Sintaxe:
	PROC SQL;
		CREATE TABLE libname.table_name AS
			SELECT columns
			FROM table
			WHERE condition
			ORDER BY columns
	QUIT;
	
	Abrir o programa p107a03.sas e seguir os passos:
	
1) Modificar a query para criar uma tabela temporária
chamada top_damage

2) Adicionar uma segunda query para gerar um report listando 
todas as colunas das primeiras 10 tempestades na tabela top_damage

3) Adicionar um título antes da segunda query para incluir
"Top 10 Storms by Damage Cost"

	a) Quantas das top 10 tempestades ocorreram em 2005?
	
***********************************************************/

ods noproctitle;
proc sql;
*Modify the query to create a table;
create table top_damage as
select Event, Date format=monyy7., Cost format=dollar16.
    from pg1.storm_damage
    order by Cost desc;
    *Add a title and query to create a top 10 report;

* Second query;
title "Top 10 Storms by Damage Cost";
SELECT *, YEAR(Date) AS Event_Year from top_damage (obs=10);
quit;