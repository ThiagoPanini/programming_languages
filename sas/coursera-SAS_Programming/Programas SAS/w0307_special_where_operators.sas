/*********************************************************
	     	     Semana 3 - Programa 7
	  Atividade 3.03 - Cláusulas especiais no WHERE
**********************************************************
1) "Descomentar" cada um dos comandos WHERE, um de cada vez, 
e executar o código para observar o resultado

2) Comentar todos os comandos WHERE anteriores. Adicionar um
filtro WHERE para printar as tempestades que começam com Z.
Quantas tempestades estão inclusas nos resultados?

***********************************************************/

proc print data=pg1.storm_summary(obs=50);
	*where MinPressure is missing; /*same as MinPressure = .*/
	*where Type is not missing; /*same as Type ne " "*/
	*where MaxWindMPH between 150 and 155;
	*where Basin like "_I";
	
	* 2) Tempestades que começam com Z;
	where Name like "Z%";
run;

/*********************************************************
Bônus: alguns operadores especiais utilizados com o WHERE:

1) Missing values:
	where var=.;
	where var=" ";
	where var is missing;
	where var is null;
	
2) Strings específicas
	% = qualquer caractere
	where City like "New%"; (New York, New Delhi, Newport, Newcastle, New)
	
	_ = caracter singular
	where City like "Sant_ %"; (Santa Clara, Santa Cruz, Santo Domingo, Santo Tomas)
	* Sant + qualquer caractere singular + espaço + qualquer coisa que vier depois

***********************************************************/