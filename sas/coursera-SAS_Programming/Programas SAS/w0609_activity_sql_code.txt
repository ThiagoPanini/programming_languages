/*********************************************************
	     	     Semana 6 - Programa 9
		Atividade 7.02 - Utilizando SQL noSAS
	
**********************************************************
	Tarefas:

1) Mostrar as colunas Event e Cost da tabela pg1.storm_damage.
Formatar valores de Cost como dollars.

2) Adicionar uma nova coluna chamada Season que extrai o 
ano da coluna Date

3) Adicionar uma cláusula WHERE para retornar apenas valores
de custo maiores que 25 bilhões

4) Adicionar cláusula ORDER BY para ordenar por Custo (DESC)

	a) Qual evento possui o maior custo?

***********************************************************/

proc sql;
	SELECT Event, 
		   Cost format=dollar13.,
		   YEAR(Date) AS Season
	FROM pg1.storm_damage
	WHERE Cost>25e9
	ORDER BY Cost DESC;
quit;