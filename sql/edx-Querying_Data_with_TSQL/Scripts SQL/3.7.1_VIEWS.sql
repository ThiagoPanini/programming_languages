--Section 3, Módulo 7, Arquivo 1 - VIEWS

--1. Introdução: Criando uma VIEW
CREATE VIEW SalesLT.vSalesOrders
AS
SELECT oh.SalesOrderID, oh.OrderDate, oh.CustomerID,
	   od.LineTotal, od.ProductID, od.OrderQty
FROM SalesLT.SalesOrderHeader oh
INNER JOIN SalesLT.SalesOrderDetail od
ON oh.SalesOrderID = od.SalesOrderID

SELECT * FROM SalesLT.vSalesOrders

--2. Criar uma View para mostrar as informações do cliente:
--CustomerID, FirstName e LastName da tabela SalesLT.Customer e
--AddressLine1, City e StateProvince da tabela SalesLT.Address
CREATE VIEW SalesLT.vCustomerAddress
AS
SELECT c.CustomerID, c.FirstName, c.LastName, 
	   a.AddressLine1, a.City, a.StateProvince
FROM SalesLT.Customer c
INNER JOIN SalesLT.CustomerAddress ca
ON c.CustomerID = ca.CustomerID
INNER JOIN SalesLT.Address a
ON ca.AddressID = a.AddressID

SELECT * FROM SalesLT.vCustomerAddress;

--3. Utilizando a View criada para unir o resultado com outra tabela
SELECT vca.StateProvince, vca.City, ISNULL(SUM(soh.TotalDue), 0.00) AS Revenue
FROM SalesLT.vCustomerAddress vca
LEFT JOIN SalesLT.SalesOrderHeader soh
ON vca.CustomerID = soh.CustomerID
GROUP BY vca.StateProvince, vca.City
ORDER BY vca.StateProvince, Revenue DESC
