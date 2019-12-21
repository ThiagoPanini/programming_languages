**************************************************;
*   Primeiro Programa em SAS - Semana 1          *;
**************************************************;
*   O objetivo deste programa é a familiarização *;
*   com o SAS Studio. Aqui, vamos:               *;
*    1) Ler os dados (data)                      *;
*    2) Visualizar os dados (proc print)         *;
*    3) Interagir com a interface SAS Studio     *;
**************************************************;

*1) Lendo os dados;
data myclass;
	set sashelp.class;
run;

proc print data=myclass;
run;