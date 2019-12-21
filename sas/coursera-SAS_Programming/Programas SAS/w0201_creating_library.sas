/*********************************************************
	     	 Semana 2 - Programa 1
	  Criando uma nova biblioteca (libname)
**********************************************************
   Pontos importantes:

   WORK library: Tabelas pessoais temporárias
   SASHELP library: Tabelas de exemplo do SAS

   WORK é a biblioteca padrão (default). Para exemplificar,
abaixo temos duas linhas de código com o mesmo resultado:

   proc contents data=work.class;
   proc contents data=class;
**********************************************************
   Demonstração:
1) Executar o código abaixo (atentando ao caminho de entra-
da fornecido, navegar nos paineis para examinar o resultado
nas bibliotecas WORK e OUT
2) Qual(is) tabelas estão presentes na lib WORK?
   Qual(is) tabelas estão presentes na lib OUT?
3) Reiniciar a sessão do SAS
4) Discutir as seguintes questões:
   a) O que ainda se encontra na lib WORK?
   b) Por que as libs OUT e PG1 não estão mais disponíveis?
   c) A tabela class_copy2 foi salva permanentemente?
   d) O que de ser feito para reestabelecer a lib OUT?
5) Para reestabelecer a lib PG1, abrir e executar o programa
libname.sas salvo previamente no diretório principal de ar-
quivos do curso
 
***********************************************************/

*Modify the path if necessary;
*Aqui, será criada uma lib de nome "out" no diretório especificado;
libname out "~/EPG194/output";

/* No bloco abaixo, será criada um dataset class_copy1 e class_copy2
que receberão o mesmo valor que a tabela class da lib sashelp*/
data class_copy1 out.class_copy2;
	set sashelp.class;
run;

proc contents data=out.class_copy2;

proc print data=out.class_copy2;