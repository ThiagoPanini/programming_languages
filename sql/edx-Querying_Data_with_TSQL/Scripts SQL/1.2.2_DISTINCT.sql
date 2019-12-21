--Section 1, Módulo 2, Arquivo 2 - DISTINCT

--1. Retornar uma lista de cores únicas da coluna Product
SELECT Color FROM SalesLT.Product;

SELECT DISTINCT COLOR, COUNT(COLOR) FROM SalesLT.Product
GROUP BY COLOR;

SELECT ISNULL(Color, 'None') FROM SalesLT.Product;
SELECT DISTINCT ISNULL(Color, 'None') AS ColorList FROM SalesLT.Product;

--2. Retornar uma lista única das colunas Color e Size
SELECT DISTINCT
	ISNULL(Color, 'None') AS Color,
	ISNULL(Size, '-') AS Size
FROM SalesLT.Product;

--3. Para este desafio, selecionar o DATABASE BrazilianECommerce
--Retornar uma lista de estados únicos (seller_state)
SELECT TOP 10 * FROM BrazilianECommerce.dbo.tb_OlistECommerce_Seller;
SELECT DISTINCT seller_state FROM BrazilianECommerce.dbo.tb_OlistECommerce_Seller;

--4. Retornar uma lista única das colunas seller_state e seller_city ordenada por estado e cidade em modo ascendente
SELECT DISTINCT seller_state, seller_city FROM BrazilianECommerce.dbo.tb_OlistECommerce_Seller ORDER BY 1;