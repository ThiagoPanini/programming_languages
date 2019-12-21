/*********************************************************
	     	     Semana 6 - Programa 1
	Atividade 6.01 - Exportando dados com PROC EXPORT
	
**********************************************************
	Sintaxe:
	PROC EXPORT DATA=input_table
			    OUTFILE="output_file"
			    <DBMS=identifier> <REPLACE>;
	
	1) Completar o comando PROC EXPORT abaixo para ler dados
da tabela pg1.storm_final e criar um arquivo csv
storm_final.csv. Utilizar &outpath para substituir o caminho
do arquivo de saída	

	a) Quantas linhas o arquivo storm_final.csv possui?

***********************************************************/

%let outpath=~/EPG194/output/;
proc export data=pg1.storm_final
			outfile="&outpath storm_final.csv"
			dbms=csv replace;
run;