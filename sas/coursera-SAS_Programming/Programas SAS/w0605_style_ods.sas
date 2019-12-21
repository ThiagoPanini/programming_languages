/*********************************************************
	     	     Semana 6 - Programa 5
	Atividade 6.03 - Exporando um arquivo Excel (Styles)
	
**********************************************************
	Abrir o programa p106a04.sas e seguir os passos:
	
1) Adicionar o comando ODS para criar um arquivo Excel 
chamado pressure.xlsx. Utilizar a variável &outpath com
o endereçamento correto. Certifique-se de fechar o ODS no fim

2) Rodar o programa e baixar o arquivo

3) Mudar o STYLE do ODS para ANALYSIS. Rodar e baixar novamente
o arquivo

***********************************************************/

%let outpath=~/EPG194/output;
ods excel file="&outpath/pressure.xlsx" style=analysis
		  options(sheet_name="Data");
		  
title "Minimum Pressure Statistics by Basin";
ods noproctitle;
proc means data=pg1.storm_final mean median min maxdec=0;
    class BasinName;
    var MinPressure;
run;

title "Correlation of Minimum Pressure and Maximum Wind";
proc sgscatter data=pg1.storm_final;
	plot minpressure*maxwindmph;
run;
title;  

ods excel close;

ods proctitle;