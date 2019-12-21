--Section 3, Módulo 7, Arquivo 5 - Common Table Expressions

--1. Exemplo
WITH CTE_year(OrderYear, CustID)
AS
(
	SELECT YEAR(OrderDate), CustomerID
	FROM SalesLT.SalesOrderHeader
)
SELECT OrderYear, COUNT(DISTINCT CustID) AS Cust_Count
FROM CTE_year
GROUP BY OrderYear;

-- -----------------------------------------------------------------------------------

--2. Utilizando Recursão com Common Table Expressions
WITH OrgReport(ManagerID, EmployeeID, EmployeeName, Level)
AS
(
	--Anchor query
	SELECT e.ManagerID, e.EmployeeID, e.EmployeeName, 0
	FROM SalesLT.Employee AS e
	WHERE ManagerID IS NULL

	UNION ALL

	--Recursive query
	SELECT e.ManagerID, e.EmployeeID, e.EmployeeName, Level+1
	FROM SalesLT.Employee e
	INNER JOIN OrgReport AS o ON e.ManagerID = o.EmployeeID
)

SELECT * FROM OrgReport
OPTION(MAXRECURSION 3);

-- ----------------------------------------------------------------------------------

--3. Outro exemplo
WITH ProductByCategory(ProductID, ProductName, Category)
AS
(
	SELECT p.ProductID, p.Name, c.Name AS Category
	FROM SalesLT.Product p
	INNER JOIN SalesLT.ProductCategory c ON p.ProductCategoryID = c.ProductCategoryID
)

SELECT Category, COUNT(ProductID) AS Products
FROM ProductByCategory
GROUP BY Category
ORDER BY Category