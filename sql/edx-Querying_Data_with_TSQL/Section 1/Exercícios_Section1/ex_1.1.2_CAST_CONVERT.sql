--Section 1, Módulo 1, Arquivo 2 - Tipos Primitivos: CAST, CONVERT

--1. Selecione as colunas ProductID e Name da tabela Product
SELECT ProductID, Name FROM SalesLT.Product

--2. Faça uma seleção concatenada com "ProductID: Name" e dê um "allias" a essa coluna: ProductName
SELECT ProductID + ': ' + Name FROM SalesLT.Product

--3. Convertendo o tópico 2 acima com CAST
--https://docs.microsoft.com/pt-br/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-2017
SELECT ProductID, Name,
	CAST(ProductID AS VARCHAR) + ': ' + Name AS ProductName
FROM SalesLT.Product

--4. Convertendo o tópico 2 acima com CONVERT
SELECT CONVERT(VARCHAR, ProductID) + ': ' + Name 
FROM SalesLT.Product

--5. Converta a coluna SellStartDate em uma data padrão e em uma data formato ISO8601 (126)


--6. Seria possível converter a coluna Size para Inteiro?


--7. Converta a coluna Size para Inteiro utilizando TRY_CAST


--8. Retorne a coluna ListPrice no formato 'R$ ListPrice'
