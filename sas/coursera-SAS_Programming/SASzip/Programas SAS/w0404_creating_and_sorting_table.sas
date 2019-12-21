/*********************************************************
	     	     Semana 4 - Programa 4
	  	Prática Level 2 - Criando uma tabela ordenada
**********************************************************
1) Criar um DATA step para ler a tabela pg1.np_species
e criar uma nova tabela chamad "fox" como uma tabela
permanente no repositório EPG194/output

2) Incluir apenas as linhas onde o atributo Category seja 
igual a "Mamma" e Common_Names conter "Fox"

3) Excluir as categorias Category, Record_Status, Occurrence e
Nativeness da tabela de output

4) Verificar que "Fox Squirrels" estão inclusos no output

5) Adicionar um filtro para eliminar linhas que contém "Squirrel"

6) Ordenar por Common_Names

	a) Qual o valor de Common_Names está na primeira linha?

***********************************************************/

libname out "~/EPG194/output";

proc print data=pg1.np_species (obs=10);
run;

data out.fox;
	set pg1.np_species;
	where Category = "Mammal" and Common_Names likes "%Fox%";
	where also Common_Names not in ('%Squirrel%');
	drop Category Record_Status Occurrence NAtiveness
run;

proc sort data=out.fox;
	by Common_Names;
run;
* Talvez "criar" uma tabela ordenada seja adicionar um statemente out=<table_name>;
*no final da proc sort;

/* Resposta: Arctic Fox */