--Section 3, Módulo 7, Arquivo 4 - Tabelas Derivadas

--1. Selecionar a Categoria e a contagem de produtos por categoria através de uma tabela derivada
SELECT Category, COUNT(ProductID) AS Products
FROM
	--Tabela derivada pode ser executada separadamente
	(SELECT p.ProductID, p.Name AS Product, c.Name AS Category
	FROM SalesLT.Product AS p
	INNER JOIN SalesLT.ProductCategory AS c
	ON p.ProductCategoryID = c.ProductCategoryID) AS ProdCats
GROUP BY Category
ORDER BY Category;