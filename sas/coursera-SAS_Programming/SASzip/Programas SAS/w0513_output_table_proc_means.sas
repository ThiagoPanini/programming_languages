/*********************************************************
	     	     Semana 5 - Programa 13
	Atividade 5.06 - Aprimorando reports com PROC MEANS
	
**********************************************************
	Abrir o programa p105a07 e seguir os passos:

1) Executar o programa sem alterações e verificar o output
gerado na tabela winds_stats. É possível encontrar as mesmas
estatísticas do report na tabela? O que as primeiras 5
linhas representam?

2) Implementar o comando WAYS. Deletar as estatísticas listadas
no PROC MEANS e adicionar uma opção NOPRINT. Note que o
report não é gerado e as primeiras 5 linhas já não mais
existem na tabela

3) Adicionar as opções a seguir no comando OUTPUT
e executar novamente o programa. Quantas linhas estão na
tabela de output.
	output out=wind_stats mean=AvgWind max=MaxWind;

***********************************************************/

* 1) Verificando primeira etapa;
proc means data=pg1.storm_final mean median max;
	var MaxWindMPH;
	class BasinName;
	*ways 1;
	output out=wind_stats;
run;
* R: As 5 primeiras linhas provavelmente representam estatísticas para a tabela inteira; 

* 2) Verificando segunda etapa;
proc means data=pg1.storm_final noprint;
	var MaxWindMPH;
	class BasinName;
	ways 1;
	output out=wind_stats;
run;

* 3) Verificando terceira etapa;
proc means data=pg1.storm_final noprint;
	var MaxWindMPH;
	class BasinName;
	ways 1;
	output out=wind_stats mean=AvgWind max=MaxWind;
run;
