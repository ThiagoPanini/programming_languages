/*********************************************************
	     	     Semana 6 - Programa 3
	Atividade 6.02
	
**********************************************************
	Abrir o programa p106a03 e seguir o passo a passo:
1) Completar o comando LIBNAME utilizando a engine XLSX para
criar um arquivo Excel com nome storm.xlsx

2) Modificar o passo DATA para escrever a tabela storm_final
no arquivo storm.xlsx. Utilizar a variável &outpath.

3) No final do passo DATA, escrever um código para limpar
a lib criada

4) Executar o código e visualizar no log que foram importados
3092 linhas no arquivo.

	a) Como as datas aparecem no arquivo storm_final?

***********************************************************/

* Criando variável para endereçamento;
%let outpath=~/EPG194/output;
libname xlout xlsx "&outpath/storm.xlsx";

data xlout.Storm_Data;
	set pg1.storm_final;
	drop Lat Lon Basin OceanCode;
run;

libname xlout clear;