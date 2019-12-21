/*********************************************************
	     	     Semana 6 - Programa 11
		Atividade 7.02 - Unindo tabelas SQL/SAS
	
**********************************************************
	Sintaxe:
	
	PROC SQL;
		SELECT columns
		FROM table1
		INNER JOIN table2
		ON table1.fk = table2.pk
		<WHERE condition>
		<ORDER BY column>;
	QUIT;


***********************************************************/

proc sql;
select class_update.Name, Grade, Age, Teacher 
    from pg1.class_update inner join pg1.class_teachers
    on class_update.Name=class_teachers.Name;
quit;  

/**********************************************************
	Demonstração:
	
1) Visualizar as colunas das tabelas pg1.storm_summary e
pg1.storm_basincodes

2) Adicionar a coluna BasinName na query antes de Basin.
O programa irá falhar sem nenhuma junção adequada entre as
tabelas.

3) Aplicar o correto JOIN entre as tabelas

***********************************************************/

proc print data=pg1.storm_summary (obs=10);
run;

proc print data=pg1.storm_basincodes (obs=10);
run;

proc sql;
	CREATE TABLE season_basin_name AS
	SELECT sm.Season,
		   sm.Name,
		   sm.Basin,
		   sb.BasinName,
		   sm.Type,
		   sm.MaxWindMPH
	FROM pg1.storm_summary sm
	INNER JOIN pg1.storm_basincodes sb
	ON sm.Basin = sb.Basin
	ORDER BY Season DESC, Name;
	
	SELECT * FROM season_basin_name (obs=10)
	WHERE Name IS NOT NULL;
quit;
	