--Section 3, Módulo 6, Arquivo 2 - Self-Contained e Correlated Subqueries

--1. Self-Contained: 
-- Retornar as colunas CustomerID, SalesOrderID, OrderDate da tabela SalesLT.Order 
-- quando tivermos a data máxima (orderdate) da tabela SalesLT.Order
SELECT CustomerID, SalesOrderID, OrderDate
FROM SalesLT.SalesOrderHeader
WHERE OrderDate = 
(SELECT MAX(OrderDate) FROM SalesLT.SalesOrderHeader)

--2. Correlated:
-- Para cada cliente, retornar o valor máximo de OrderDate
SELECT CustomerID, SalesOrderID, OrderDate 
FROM SalesLT.SalesOrderHeader SO1
WHERE OrderDate = 
	(SELECT MAX(OrderDate) 
	FROM SalesLT.SalesOrderHeader SO2
	WHERE SO1.CustomerID = SO2.CustomerID)

-- Antes do update:
--	CustomerID: 29847
--	SalesOrderID: 71774
--	OrderDate: 2004-06-01 00:00:00.000

-- Depois do update:
--	CustomerID: 29847
--	SalesOrderID: 71774
--	OrderDate: 2016-12-25 11:05:36.587
UPDATE SalesLT.SalesOrderHeader SET OrderDate = GETDATE()-1000 WHERE CustomerID = 29847;