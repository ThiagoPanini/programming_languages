**************************************************;
*   PDF Practicing with SAS Studio - Semana 1    *;
**************************************************;
*   O objetivo deste programa é a familiarização *;
*   com o SAS Studio. Aqui, vamos:               *;
*    1) Ler os dados (data)                      *;
*    2) Explorar e visualizar dados (proc means) *;
*    3) Interagir com a interface SAS Studio     *;
**************************************************;

* Lendo os dados;
data work.shoes;
	set sashelp.shoes;
	NetSales = Sales - Returns; *Criando nova coluna;
run;

* Agregando dados em uma visão de média;
proc means data=work.shoes mean sum maxdec=2; 
	*Acima, a definição de agregração por média e soma (2 casas decimais);
	*Abaixo, a definição das variáveis;
	var NetSales; 
	class region;
run;