/*********************************************************
	     	     Semana 4 - Programa 8
	          Demo - Caractere Functions
**********************************************************
Sintaxe:
	DATA output_table;
		SET input_table;
		new_column = function(arguments);
	RUN;
	
Numeric Functions:
	SUM(num1, num2, ...)
	MEAN(num1, num2, ...)
	MEDIAN(num1, num2, ...)
	RANGE(num1, num2, ...)
	
Caractere Functions:
	UPCASE(char)
	PROPCASE(char, <delimiters>)
	CATS(char1, char2)
	SUBSTR(char, position, <length>)

***********************************************************/

* Criando nova tabela;
data storm_new;
	set pg1.storm_summary;
	drop Type Hem_EW Hem_NS MinPressure Lat Lon;
	Basin = upcase(Basin); *transforma a coluna Basin em letra maiúscula;
	Name = propcase(Name); *transforma a coluna Name em capitalize;
	Hemisphere = cats(Hem_NS, Hem_EW); *concatena as duas colunas Hem;
	Ocean = substr(Basin, 2, 1); *retira o código do oceano da coluna Basin;
run;