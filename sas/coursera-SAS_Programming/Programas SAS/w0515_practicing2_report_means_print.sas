/*********************************************************
	     	     Semana 5 - Programa 15
	Prática Level 2 - Criando report com PROC MEANS
	
**********************************************************
	A tabela pg1.np_westweather contém dados climáticos de
quatro parques nacionais: Death Valley National Park,
Grand Canyon National Park, Yellowstone National Park e
Zion National Park. Utilizar o PROC MEANS para analisar
dados estatísticos dessa tabela de acordo com as seguintes
especificações:

1) Eliminar linhas onde o valor de Precip for igual a 0

2) Analisar precipitação agrupados por Name e Year

3) Criar uma tabela de output de nome rainstats com colunas
de N (contagem) e soma

4) Nomear as colunas como RainDays e TotalRain, respectivamente

5) Manter apenas as linhas de combinação de Year e Name

	a) Quantas linhas existem em work.rainstats?

***********************************************************/

proc means data=pg1.np_westweather noprint;
	var Precip;
	class Name Year;
	where Precip ne 0;
	ways 2;
	output out=rainstats n=RainDays sum=TotalRain;
run;

/**********************************************************
	Parte 2:

1) Executar um PROC PRINT na tabela rainstats

2) Printar na ordem Name, Year, RainDays e TotalRain

3) Etiquetar colunas
	Name: Park Name
	RainDays: Number of Days Raining
	Total Rain: Total Rain Amount (inches)
	
4) Título "Rain Statistics by Year and Park"

	a) Qual o valor da coluna Total Rain Amount (inches)
na primeira linha?

***********************************************************/

title "Rain Statistics by Year and Park";
proc print data=rainstats label noobs;
	var Name Year RainDays TotalRain;
	label Name="Park Name"
		  RainDays="Number of Days Raining"
		  TotalRain="Total Rain Amount (inches)";
run;
title;
