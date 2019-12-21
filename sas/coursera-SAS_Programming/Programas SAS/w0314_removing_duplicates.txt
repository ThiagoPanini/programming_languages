/*********************************************************
	     	     Semana 3 - Programa 14
	  	       Removendo duplicatas no SAS
**********************************************************
Sintaxe:

    PROC SORT DATA=input_table <OUT=output_table>
    	NODUPRECS <DUPOUT=output_table>;
    BY _ALL;
    RUN;

***********************************************************/

proc sort data=pg1.class_test3
	out=test_clean
	noduprecs
	dupout=test_dups;
	by _all_;
run;