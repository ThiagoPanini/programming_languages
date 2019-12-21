--Section 2, Módulo 5, Arquivo 1 - SCALAR FUNCTIONS

--1. Funções de data: YEAR, MONTH, DAY
SELECT TOP 10 * FROM SalesLT.Product;
SELECT YEAR(SellStartDate) AS AnoInicioVenda, ProductID, Name FROM SalesLT.Product ORDER BY AnoInicioVenda;

--2. Funções de data: DATENAME
SELECT 
	ProductID, 
	Name,
	SellStartDate AS DataVenda,
	YEAR(SellStartDate) AS AnoInicioVenda,
	MONTH(SellStartDate) AS MesInicioVenda,
	DAY(SellStartDate) AS DiaInicioVenda,
	DATENAME(mm, SellStartDate) AS NomeMes,
	DATENAME(dw, SellStartDate) AS DiaDaSemana
FROM SalesLT.Product

--3. Funções de data: DATEDIFF e GETDATE()
SELECT 
	ProductID, 
	Name,
	SellStartDate AS DataVenda,
	DATEDIFF(yy, SellStartDate, GETDATE()) AS IdadeEmAnos
FROM SalesLT.Product

--4. Strings: UPPER e LOWER
SELECT UPPER(Name) AS UpperName, LOWER(Name) AS LowerName
FROM SalesLT.Product

--5. Strings: CONCAT
SELECT TOP 10 * FROM SalesLT.Customer;
SELECT 
	FirstName AS Nome,
	LastName AS Sobrenome,
	CONCAT(FirstName + ' ', LastName) AS NomeCompleto 
FROM SalesLT.Customer

--6. Strings: LEFT
SELECT ProductNumber, LEFT(ProductNumber, 2) AS TipoProduto 
FROM SalesLT.Product

--7. Strings: SUBSTRING e CHARINDEX
SELECT 
	Name AS NomeProduto,
	ProductNumber AS NumeroProduto,
	LEFT(ProductNumber, 2) AS TipoProduto,
	SUBSTRING(ProductNumber, CHARINDEX('-', ProductNumber) + 1, 4) AS CódigoModelo
FROM SalesLT.Product

--7.X Strings: SUBSTRING e CHARINDEX (inverso da string)
SELECT 
	Name AS NomeProduto,
	ProductNumber AS NumeroProduto,
	LEFT(ProductNumber, 2) AS TipoProduto,
	SUBSTRING(ProductNumber, CHARINDEX('-', ProductNumber) + 1, 4) AS CódigoModelo,
	SUBSTRING(ProductNumber, LEN(ProductNumber) - CHARINDEX('-', REVERSE(RIGHT(ProductNumber, 3))) + 2, 2) AS CódigoTamanho
FROM SalesLT.Product

--Explicando
SELECT 
	ProductNumber AS NumeroProduto,
	RIGHT(ProductNumber, 3) AS DireitaOriginal,
	REVERSE(ProductNumber) AS ReverseProdNumber,
	REVERSE(RIGHT(ProductNumber, 3)) AS DireitaReverse,
	LEN(ProductNumber) AS ComprimentoProdNumber,
	CHARINDEX('-', REVERSE(RIGHT(ProductNumber, 3))) + 2 AS CharidxArg,
	LEN(ProductNumber) - CHARINDEX('-', REVERSE(RIGHT(ProductNumber, 3))) + 2 AS SecondArgSubstring,
	SUBSTRING(ProductNumber, LEN(ProductNumber) - CHARINDEX('-', REVERSE(RIGHT(ProductNumber, 3))) + 2, 2) AS CódigoTamanho
FROM SalesLT.Product

