/*********************************************************
	     	     Semana 4 - Programa 3
	  	Prática Level 1 - Criando uma tabela SAS
**********************************************************
1) Abrir a tabela pg1.eu_occ e examinar os dados

2) Abrir o documento p104p01.sas da pasta "practices"

3) Modificar o código para criar uma tabela temporária
chamada eu_occ2016 e ler a tabela pg1.eu_occ

4) Completar o comando WHERE para selecionar apenas instâncias
reportadas em 2016. Notar que a variável YearMon é uma coluna
de caractere onde os 4 primeiros elementos representam o ano

5) Completar o comando FORMAT para aplicar o formato COMMA17.
às variáveis Hotel, ShortStay e Camp

6) Completar o comando DROP para excluir a coluna Geo 

	a) Quantas linhas presentes na tabela eu_occ2016?

***********************************************************/

/* 1) Verificando conteúdo da tabela pg1.eu_occ */
proc print data=pg1.eu_occ (obs=10);
run;

/* 2) Abrir Documento */
/* 3) Criar uma tabela eu_occ2016 */
data eu_occ2016;
	set pg1.eu_occ;
	where YearMon like '2016%';
	format Hote ShortStay Camp comma7.;
	drop Geo;
run;