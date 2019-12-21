/*********************************************************
	     	     Semana 5 - Programa 14
	Prática Level 1 - Criando report com PROC MEANS
	
**********************************************************
	A tabela pg1.np_westweather contém dados climáticos de
quatro parques nacionais: Death Valley National Park,
Grand Canyon National Park, Yellowstone National Park e
Zion National Park. Utilizar o PROC MEANS para analisar
dados estatísticos dessa tabela de acordo com as seguintes
especificações:

1) Visualizar média, mínimo e máximo para os atributos
Precip, Snow, TempMin e TempMax

2) Utilizar MAXDEC= para arredondar valores em 2 casas decimais

3) Utilizar CLASS para agrupar dados por Year e Name

4) Título "Weather Statistics by Year and Park"

	a) Qual é a média de TempMin no parque Death VAlley em 2016?

***********************************************************/

ods noproctitle;
/*proc print data=pg1.np_westweather (obs=5);
run;*/

title "Weather Statistics by Year and Park";
proc means data=pg1.np_westweather mean min max maxdec=2;
	var Precip Snow TempMin TempMax;
	class Year Name;
run;
ods proctitle;
title;