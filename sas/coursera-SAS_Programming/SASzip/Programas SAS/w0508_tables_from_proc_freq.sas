/*********************************************************
	     	     Semana 5 - Programa 8
	 Atividade 5.04 - Customizando tabelas e reports
**********************************************************
1) Criar uma tabela de nome storm_count completando a
opção OUT= na linha TABLES
	a) Quais valores foram inclusos na tabela e quais
estatísticas foram calculadas?

2) Colocar as variáveis StartDate e BasinName em códigos
TABLE separados. Adicionar uma opção OUT= em cada comando
e criar uma tabela para cada (month_count para StartDate e
basin_count para BasinName)
	a) Qual mês teve o maior número de tempestades?

***********************************************************/

title "Frequency Report for Basin and Storm Month";

*1);
proc freq data=pg1.storm_final order=freq noprint;
	tables StartDate BasinName / out=storm_count; *Somente BasinName será incluso (listado por último);
	format StartDate monname.;
run;

*2);
proc freq data=pg1.storm_final order=freq noprint;
	tables StartDate  / out=month_count; *Aqui teremos uma tabela para cada análise;
	tables BasinName / out=basin_count;
	format StartDate monname.;
run;
