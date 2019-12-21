--Section 4, Módulo 10, Arquivo 4 - STORED PROCEDURE

/*****************************************************************
					STORED PROCEDURE

	As Stored Procedures são criadas para encapsular o código e
definir um bloco responsável por executar determinada ação definida
neste bloco. Stored Procedures podem ter inputs e também podem
retornar valores como outputs.

			CREATE PROCEDURE <procedure_name>(<params>)
			AS
			<procedure_code>

	Para executar uma procedure: EXEC <procedure_name>
*****************************************************************/

--1. Criando uma Procedure
CREATE PROCEDURE SalesLT.GetProductsByCategory (@CategoryID INT = NULL)
AS
IF @CategoryID IS NULL
	SELECT ProductID, Name, Color, Size, ListPrice
	FROM SalesLT.Product
ELSE
	SELECT ProductID, Name, Color, Size, ListPrice
	FROM SalesLT.Product
	WHERE ProductCategoryID = @CategoryID;

--1.1 Executando Procedure (sem parâmetros de entrada -> ProductCategoryID = NULL)
EXEC SalesLT.GetProductsByCategory;

--1.2 Executando Procedure (com parâmetro de entrada)
EXEC SalesLT.GetProductsByCategory 6;