--Section 2, Módulo 3, Arquivo 2 - OUTER JOIN

--1. Retornar clientes (FirstName e LastName) da tabela Customer que fizeram pedidos (SalesOrderNumber da tabela SalesOrderHeader)
SELECT TOP 10 * FROM SalesLT.Customer;
SELECT TOP 10 * FROM SalesLT.SalesOrderHeader;

SELECT c.FirstName, c.LastName, oh.SalesOrderNumber
FROM SalesLT.Customer c
INNER JOIN SalesLT.SalesOrderHeader oh
ON c.CustomerID = oh.CustomerID

--2. Retornar todos os clientes e os pedidos, independente se este realizou algum ou não
SELECT c.FirstName, c.LastName, oh.SalesOrderNumber
FROM SalesLT.Customer c
LEFT JOIN SalesLT.SalesOrderHeader oh
ON c.CustomerID = oh.CustomerID

--3. Retornar apenas clientes que não compraram nada
SELECT c.FirstName, c.LastName, oh.SalesOrderNumber
FROM SalesLT.Customer c
LEFT JOIN SalesLT.SalesOrderHeader oh
ON c.CustomerID = oh.CustomerID
WHERE oh.SalesOrderID IS NULL