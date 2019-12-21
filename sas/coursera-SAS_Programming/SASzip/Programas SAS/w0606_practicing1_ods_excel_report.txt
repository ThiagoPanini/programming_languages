/*********************************************************
	     	     Semana 6 - Programa 6
	Prática Level 1 - Criando Arquivo Excel com ODS
	
**********************************************************
	Abrir o programa p106p01.sas e seguir os passos:
	
1) Escrever o resultado em "&outpath/StormStats.xlsx" com
STYLE=snow e SHEET_NAME=South Pacific Summary para a primeira
aba

2) Imediatamente após o passo PROC PRINT, adicionar uma 
opção no ODS mudando o nome da aba para Detail

***********************************************************/
%let outpath=~/EPG194/output;
ods excel file="&outpath/StormStats.xlsx"
          style=snow
          options(sheet_name="South Pacific Summary");
          
proc means data=pg1.storm_detail maxdec=0 median max;
    class Season;
    var Wind;
    where Basin='SP' and Season in (2014,2015,2016);
run;

ods excel options(sheet_name="Detail");
proc print data=pg1.storm_detail noobs;
    where Basin='SP' and Season in (2014,2015,2016);
    by Season;
run;

ods excel close;
