--Section 2, Módulo 5, Arquivo 3 - WINDOW FUNCTIONS

--1. Ranqueando produtos a partir do preço
SELECT TOP 100 ProductID, Name, ListPrice,
	RANK() OVER(ORDER BY ListPrice DESC) AS RankByPrice
FROM SalesLT.Product
ORDER BY RankByPrice

--2. Ranqueando produtos por cada categoria
SELECT c.Name AS Categoria, p.Name AS Produto, ListPrice,
	RANK() OVER(PARTITION BY c.Name ORDER BY ListPrice DESC) AS RankByPrice
FROM SalesLT.Product p
INNER JOIN SalesLT.ProductCategory c
ON p.ProductCategoryID = c.ProductCategoryID
ORDER BY Categoria, RankByPrice;