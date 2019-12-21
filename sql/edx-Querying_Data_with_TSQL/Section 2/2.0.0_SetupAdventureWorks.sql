-- Criando uma View
CREATE VIEW [SalesLT].[Customers] AS
SELECT DISTINCT FirstName, LastName
FROM SalesLT.Customer
WHERE LastName >='m'
OR CustomerID = 3;
GO

CREATE VIEW [SalesLT].[Employees] AS
SELECT DISTINCT FirstName, LastName
FROM SalesLT.Customer
WHERE LastName <='m'
OR CustomerID = 3;
GO

SELECT FirstName, LastName
FROM SalesLT.Employees
UNION
SELECT FirstName, LastName
FROM SalesLT.Customers
ORDER BY LastName;