--Section 4, Módulo 9, Arquivo 2 - UPDATE

/*****************************************************************
							UPDATE

	Atualização de elementos da tabela a partir de condições
pré-definidas na cláusula WHERE. Sintaxe:

UPDATE <table> SET <column(s)> = <value(s)> 
WHERE <condition>
*****************************************************************/

/*****************************************************************
							MERGE

	O comando MERGE modifica a base de dados em condições:
	- Quando a tabela source bate com a tabela target
	- Quando a tabela source não tem vínculo com a target
	- Quando a tabela target não tem vínculo com a fonte

MERGE INTO <table_source> AS <allias>
	USING <table_target> AS <allias2>
	ON <allias1>.<column> = <allias2>.<column>
WHEN MATCHED THEN
	UPDATE SET <column> = <value>
WHERE NOT MATCHED THEN
	INSERT (<columns>)
	VALUES (<columns<);
*****************************************************************/

/*****************************************************************
							DELETE

	Comando que simplesmente elimina um ou mais registros da 
tabela no Banco de Dados. Sintaxe:

DELETE FROM <table> WHERE <condition>
*****************************************************************/

/*****************************************************************
					    TRUNCATE TABLE

	O comando TRUNCATE limpa a tabela inteira de uma única vez. 
As linhas são eliminadas simultaneamente e não individualmente
*****************************************************************/

/* -------------------------------------------------------------------------------------------------*/

--0. Visualizando os dados a serem utilizados como exemplo
SELECT * FROM SalesLT.CallLog;

--1. Atualizando dados nulos
UPDATE SalesLT.CallLog SET Notes = 'No notes' WHERE Notes IS NULL;

--1.1 Atualizando múltiplas colunas
UPDATE SalesLT.CallLog SET SalesPerson = '', PhoneNumber = '';

--1.2 Atualizando dados a partir do retorno de uma query
UPDATE SalesLT.CallLog 
SET SalesPerson = c.SalesPerson, PhoneNumber = c.Phone
FROM SalesLT.Customer c
WHERE c.CustomerID = SalesLT.CallLog.CustomerID;

/* -------------------------------------------------------------------------------------------------*/

--2. Deletando linhas (primeira linha)
DELETE FROM SalesLT.CallLog
WHERE CallTime < DATEADD(dd, -7, GETDATE())

SELECT * FROM SalesLT.CallLog