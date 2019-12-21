/*********************************************************
	     	     Semana 5 - Programa 10
			Prática Level 1 - Proc FREQ
**********************************************************
	Utilizar a PROC FREQ na tabela pg1.np_species para:
	
	1) Utilizar o comando TABLES para gerar um report com
o atributo Category
	2) Utilizar o comando NOCUM para suprimir colunas cumulativas
	3) Utilizar ORDER=FREQ parar ordenar os resultados por
frequência
	4) Título: Categories of Reported Species
	
	a) Qual o percentual da espécie Fungi?

***********************************************************/

ods noproctitle;
title "Categories of Reported Species";
proc freq data=pg1.np_species order=freq;
	tables Category / nocum;
run;
title;
ods proctitle;

/**********************************************************
	Modificar a PROC FREQ para contemplar as alterações:
	
	1) Incluir apenas linhas onde Species_ID iniciar com "EVER"
e Category não for "Vascular Plant". 
	Nota: "EVER" representa Everglades National Park

	2) Acionar a opção ods graphics antes da proc freq e
"desligar" títulos da procedure

	3) Adicionar plots=freqplot para análise gráfica
	
	4) Adicionar Everglades como segundo título
	
	a) Qual o valor da Categoria com menor frequência?
	
***********************************************************/

ods noproctitle;
ods graphics on;
title "Categories of Reported Species";
title2 "Everglades";
proc freq data=pg1.np_species order=freq;
	tables Category / nocum plots=freqplot;
	where Species_ID like "EVER%" and
	      Category ne "Vascular Plant";
run;
ods proctitle;
title;