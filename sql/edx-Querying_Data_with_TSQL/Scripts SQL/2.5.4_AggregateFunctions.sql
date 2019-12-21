--Section 2, Módulo 5, Arquivo 3 - AGGREGATE FUNCTIONS

--1. Retornando a contagem total de produtos, a contagem apenas de categorias distintas e a média do preço de lista
SELECT
	COUNT(*) AS Produtos,
	COUNT(DISTINCT(ProductCategoryID)) AS CategoriasDistintas,
	AVG(ListPrice) AS ValorMedio
FROM SalesLT.Product

--2. Retornar o valor total vendido para cada um dos vendedores. SalesLT.Customer.SalesPerson 
--(unindo, através de um LEFT JOIN, a tabela SalesLT.SalesOrderHeader)
SELECT c.SalesPerson, CAST(ISNULL(SUM(oh.Subtotal), 0.00) AS DECIMAL(9, 2)) AS Vendas
FROM SalesLT.Customer c
LEFT JOIN SalesLT.SalesOrderHeader oh
ON c.CustomerID = oh.CustomerID
GROUP BY c.SalesPerson
ORDER BY Vendas DESC;

--3. Alterando um atributo a mais na consulta anterior para agrupamento
SELECT 
	c.SalesPerson,
	CONCAT(c.FirstName + ' ', c.LastName) AS Cliente,
	CAST(ISNULL(SUM(oh.Subtotal), 0.00) AS DECIMAL(9, 2)) AS Vendas
FROM SalesLT.Customer c
LEFT JOIN SalesLT.SalesOrderHeader oh
ON c.CustomerID = oh.CustomerID
GROUP BY c.SalesPerson, CONCAT(c.FirstName + ' ', c.LastName)
ORDER BY Vendas DESC;