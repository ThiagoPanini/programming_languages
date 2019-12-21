--Section 3, Módulo 6, Arquivo 1 - Scalar e Multi-Valued Subqueries

--1. Subqueries escalares
SELECT SalesOrderID, ProductID, UnitPrice, OrderQty
FROM SalesLT.SalesOrderDetail
WHERE SalesOrderID = 
	(SELECT MAX(SalesOrderID) AS LastOrder
	FROM SalesLT.SalesOrderDetail)

--2. Da tabela SalesLT.Products, retornar todas as linhas onde a coluna ListPrice for maior 
--que o valor máximo de UnitPrice da tabela SalesLT.SalesOrderDetail

--2.1 Primeiro passo: Verificando o valor máximo de UnitPrice
SELECT MAX(UnitPrice) FROM SalesLT.SalesOrderDetail;

--2.2 Segundo Passo: Entrando manualmente com o valor obtido
SELECT * FROM SalesLT.Product WHERE ListPrice > 1466.01;

SELECT * FROM SalesLT.Product
WHERE ListPrice > 
	(SELECT MAX(UnitPrice) FROM SalesLT.SalesOrderDetail)

--3. Subqueries multi valoradas
SELECT CustomerID, SalesOrderID
FROM SalesLT.SalesOrderHeader
WHERE CustomerID IN (
	SELECT CustomerID
	FROM SalesLT.Customer
	WHERE Title = 'Mr.')

--4. Selecionar as colunas ProductID e UnitPrice da tabela SalesLT.SalesOrderDetail de produtos 
--cuja coluna ProductNumber da tabela SalesLT.Product inicie com "FR"
SELECT ProductID, UnitPrice FROM SalesLT.SalesOrderDetail
WHERE ProductID IN
	(SELECT ProductID FROM SalesLT.Product
	WHERE ProductNumber LIKE 'FR%')

-- Uma outra forma de obter o mesmo resultado
SELECT od.ProductID, od.UnitPrice
FROM SalesLT.SalesOrderDetail od
INNER JOIN SalesLT.Product prod
ON od.ProductID = prod.ProductID
WHERE prod.ProductNumber LIKE 'FR%'

