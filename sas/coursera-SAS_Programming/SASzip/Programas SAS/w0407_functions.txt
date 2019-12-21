/*********************************************************
	     	     Semana 4 - Programa 7
	Atividade 4.05 - Criando novas colunas com Funções
**********************************************************
1) Examinar as colunas da tabela pg1.storm_range
Notar que há quatro medições de Wind para cada tempestade

2) Criar uma nova coluna chamada WindAvg com a média das 4

3) Criar uma coluna WindRange com o range entre as 4

	a) Qual a medição de WindRange da primeira linha?
***********************************************************/

data wind_analysis;
	set pg1.storm_range;
	WindAVG = MEAN(Wind1, Wind2, Wind3, Wind4);
	*WindRange = MAX(Wind1, Wind2, Wind3, Wind4) - MIN(Wind1, Wind2, Wind3, Wind4);
	WindRange = RANGE(Win1, Wind2, Wind3, Wind4);
run;