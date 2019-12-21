/*********************************************************
	     	     Semana 2 - Programa 2
	  Atividade 2.05 - Library para ler dados em Excel
**********************************************************
   Conteúdo do programa:

1) Criação de biblioteca para leitura de dataset em Excel
   a) comando VALIDVARNAME=V7 para nomes das colunas
   b) comando LIBNAME libref engine path para ler dados
2) Verificação de conteúdo lido (proc datasets lib=libref)
3) Visualização de uma das tabelas (proc data=libref.table)
 
***********************************************************/
*Criando biblioteca NP para ler arquivo Excel;
options validvarname=v7;
libname np xlsx "~/EPG194/data/np_info.xlsx";

*Verificando datasets contidos na biblioteca;
proc datasets lib=np;

*Verificando conteúdo do dataset Parks;
proc contents data=np.parks;
run;
/*Aqui, é possível perceber que algumas colunas da tabela Parks
possuem espaços em seus nomes. Para evitar isso, foi adicionado
um comando específico dentro do statement OPTIONS*/

*Limpando biblioteca;
libname np clear;

 