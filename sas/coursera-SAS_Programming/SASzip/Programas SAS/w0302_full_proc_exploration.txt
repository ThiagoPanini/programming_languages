/*********************************************************
	     	     Semana 3 - Programa 2
	  Prática 1 - Explorando dados com comando proc
**********************************************************
Objetivos:

1) Completar o comando "proc print" para mostrar as 20
primeiras observações da tabela pg1.np_summary

2) Adicionar o comando "var" para incluir apenas as variáveis:
Reg, Type, ParName, DayVisits, TentCampers and RVCampers.
	a) É possível observar algumas inconsistência nos dados?
	
3) Copiar, colar e alterar o comando anterior, passando de
"proc print" para "proc means" e retirando o parâmetro limita-
dor de observações. Modificar o parâmetro "var" para mostrar
estatísticas das variáveis DayVisits, TentCampers e RVCampers.
	a) Qual é o valor mínimo para TentCampers?
	b) Trata-se de um valor inesperado?
	
4) Copiar, colar e alterar o comando anterior, passando de 
"proc means" para "proc univariate".
	a) Existem valores negativos nas colunas?
	
5) Copiar, colar e alterar o comando anterior, passando de
"proc univariate" para "proc freq". Modificar o parâmetro
"var" para "tables", assimilando as variáveis Reg e Type.
	a) Existem códigos em lowecase?
	b) Existe algum código que ocorre apenas uma vez na tabela?
	
6) Documentar cada procedimento e salvar o programa final
como "np.validate.sas" na pasta "output"
 
***********************************************************/

/* 1) proc print: 20 primeiras observações */
proc print data=pg1.np_summary (obs=20);
run;

/* 2) proc print: restringindo colunas */
proc print data=pg1.np_summary (obs=20);
	var Reg Type ParkName DayVisits;
run;

/* 3) proc means: estatística */
proc means data=pg1.np_summary;
	var DayVisits TentCampers RVCampers;
run;
/* a) O valor mínimo para TentCampers é 0.
   b) Não é um valor esperado, pois essa variável identifica a quantidade
de campistas do Camping. */

/* 4) proc univariate: valores extremos */
proc univariate data=pg1.np_summary;
	var DayVisits TentCampers RVCampers;
run;

/* 5) proc freq: contagem de atributos */
proc freq data=pg1.np_summary;
	tables Reg Type;
run;
/* a) Não há códigos em lowercase */