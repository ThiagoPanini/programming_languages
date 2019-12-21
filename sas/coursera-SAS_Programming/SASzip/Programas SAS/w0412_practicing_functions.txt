/*********************************************************
	     	     Semana 4 - Programa 12
	     Prática Level 2 - Novas funções de Data
**********************************************************
A tabela pg1.eu_occ contém dados sobre noites passadas em
hoteis, estadias, acomodações para cada mês e ano. A coluna
YearMon é do tipo string.

1) Criar um passo para ler a tabela pg1.eu_occ e armazenar
os dados em uma tabela temporária chamada eu_occ_total

2) Criar as seguintes colunas:
	Year: ano (4 dígitos) extraídos da coluna YearMon
	Month: mês (2 dígitos) extraídos da coluna YearMon
	ReportDate: O primeiro dia do report mensal (utilizar) a
função MDY e as novas colunas Year e Month criadas
	Total: o total de noites para cada estabelecimento
	
3) Formatar as colunas Hotel, ShortStay, Camp e Total com 
vírgulas. Formatar ReportDate para mostrar os valores no
formato 01JAN2018

4) Manter as colunas Country, Hotel, ShortStay, Camp,
ReportDate e Total na nova tabela
***********************************************************/

* Verificando tabela;
proc print data=pg1.eu_occ (obs=10);
run;

/* Criando tabela e variáveis */
data eu_occ_total;
	set pg1.eu_occ;
	Year=SUBSTR(YearMon, 1, 4);
	Month=SUBSTR(YearMon, 6, 2);
	ReportDate=MDY(Month, 01, Year);
	Total=SUM(Hotel, ShortStay, Camp);
	
	Format ReportDate MONYY7. 
		   Hotel Total comma10.
		   ShortStay comma9.
		   Camp comma8.;
		   
	keep Country Hotel ShortStay Camp ReportDate Total;
run;

proc print data=eu_occ_total (obs=10);
run;
