/*********************************************************
	     	     Semana 3 - Programa 12
	  	   Prática Level 2 - Macro Variables
**********************************************************
1) Abrir uma nova janela e executar uma proc freq para 
analisar a tabela pg1.np_species. Gerar tabelas de frequência
para os atributos Abundance e Conservation_Status

2) Na saída da proc freq, incluir apenas linhas onde a var
Species_ID começar com "YOSE" (Yosemite National Park) e
Categoria igual a "Mammals"

3) Escreve um proc print para listar o mesmo subset anterior

4) Incluir as variáveis Species_ID, Category, Scientific_Name
e Common_Names no report

a) Quantas linhas atendem as condições anteriores?

***********************************************************
						Parte 2
**********************************************************
1) Criar uma macro variable chamada ParkCode para armazenar
a string "YOSE" e uma outra macro variable chamada SpeciesCat
para armazenar a string "Mammal"

2) Modificar o código para considerar as variáveis

3) Executar o código e confirmar os mesmos resultados

4) Mudar a variávei para "ZION" e "Bird" respectivamente e
executar novamente

***********************************************************/

/* Parte 1 */
proc freq data=pg1.np_species;
	tables Abundance Conservation_Status;
	where Species_ID like "YOSE%" and Category="Mammal";
run;

proc print data=pg1.np_species;
	var Abundance Conservation_Status Species_ID Category Scientific_Name
Common_Names;
	where Species_ID like "YOSE%" and Category="Mammal";
run;

/* Parte 2 */
%let ParkName=ZION;
%let SpeciesCat=Bird;
proc freq data=pg1.np_species;
	tables Abundance Conservation_Status;
	where Species_ID like "&ParkName%" and Category="&SpeciesCat";
run;

proc print data=pg1.np_species;
	var Abundance Conservation_Status Species_ID Category Scientific_Name
Common_Names;
	where Species_ID like "&ParkName%" and Category="&SpeciesCat";
run;

