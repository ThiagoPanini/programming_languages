/*********************************************************
	     	     Semana 2 - Programa 3
	   Atividade 2.06 - Lendo arquivo tabular
**********************************************************
   Conteúdo do programa:

1) Importando arquivo tabular (proc import)
   a) opções: datafile, dbms, out
2) Adicionar opção "replace" para poder rodar o programa
novamente
 
***********************************************************/

*Lendo dados;
proc import datafile="~/EPG194/data/storm_damage.tab"
			dbms=tab out=storm_damage_tab replace;
run;