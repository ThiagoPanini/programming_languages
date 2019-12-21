/*********************************************************
	     	     Semana 6 - Programa 8
				Demonstração - PROC SQL
	
**********************************************************
	Sintaxe:
	
	PROC SQL;
		SELECT col_name, col_name FORMAT=fmt
		FROM input_table
		WHERE expression
		ORDER BY col_name <DESC>;
	QUIT;

***********************************************************/

proc sql;
select Name, Age, Height*2.54 as HeightCM format 5.1,
       Birthdate format=date9.
    from pg1.class_birthdate
    where age > 14
    order by Height desc;
quit;

/**********************************************************
	Seguir os passos:

1) Adicionar um comando SELECT para retornar todas as colunas
databela pg1.storm_final.

2) Modificar a query para retornar apenas Season, Name,
StartDate e MaxWindMPH. Formatar StartDate como MMDDYY10.

3) Modificar Name no SELECT para converter valores para
PROPCASE

4) Adicionar cláusula WHERE para incluir storms durante
ou depois 2000 com MaxWindMPH maior que 156

5) Adicionar cláusula ORDER BY para ordenar valores por
MaxWindMPH descendente e, após isso, por Name

6) Adicionar comando TITLE para descrever o report

***********************************************************/

ods noproctitle;
title "Querying Storm Data";
proc sql;
	SELECT Season, PROPCASE(Name), StartDate format=MMDDYY10., MaxWindMPH 
	FROM pg1.storm_final
	WHERE Season>2000 and MaxWindMPH>156
	ORDER BY MaxWindMPH DESC, Name;
*Add SELECT statement;

quit;
ods proctitle;
title;
