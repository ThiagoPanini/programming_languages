/*********************************************************
	     	     Semana 3 - Programa 11
	  	   Prática Level 1 - Filtrando linhas
**********************************************************
A tabela pg1.np_summary contém dados do National Park Service.
Os códigos de parque estão inconsistentes para parques de
preservação. Examine essas inconsistências produzindo um report
que liste qualquer parque nessas condições.

1) Adicionar uma cláusula WHERE para printar apenas linhas onde
o atributo ParkName contém preserve.

2) Execute o programa.
	a) Quais códigos são utilizados para parques de presevação?

***********************************************************/

proc print data=pg1.np_summary;
	where ParkName like "%Preserve%";
run;

/* a) PRE, PRESERVE, NPRE */