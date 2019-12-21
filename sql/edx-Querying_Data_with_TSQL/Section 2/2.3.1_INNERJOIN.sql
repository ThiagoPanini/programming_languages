--Section 2, Módulo 3, Arquivo 1 - INNER JOIN

--Retornar as TOP 10 categorias com maior totalização de vendas

SELECT TOP 10 * 
FROM BrazilianECommerce.dbo.tb_OlistECommerce_Products

select top 10 * from 
BrazilianECommerce.dbo.tb_OlistECommerce_OrdersItems

--OrderItems
--order_id, -order_item_id, -product_id, -seller_id, -price, -freight_

SELECT top 10
	Prod.category_name AS NomeCategoria,
	'R$ ' + CAST(CAST(SUM(Ord.price) AS MONEY) AS VARCHAR) ValorVenda
FROM BrazilianECommerce.dbo.tb_OlistECommerce_Products AS Prod
INNER JOIN BrazilianECommerce.dbo.tb_OlistECommerce_OrdersItems AS Ord
ON Prod.product_id = Ord.product_id
GROUP BY Prod.category_name
ORDER BY SUM(Ord.price) DESC

beleza_saude	R$ 1258681.34
relogios_presentes	R$ 1205005.68
cama_mesa_banho	R$ 1036988.68









--Objetivo INNER JOIN
SELECT TOP 10 
	Prod.category_name AS Categoria, 
	SUM(Item.price) AS Total_Vendas
FROM tb_OlistECommerce_Products Prod
INNER JOIN tb_OlistECommerce_OrdersItems Item
ON Prod.product_id = Item.product_id
GROUP BY Prod.category_name
ORDER BY 2 DESC

--Desafio INNER JOIN
SELECT TOP 50
	'R$' + CAST(CAST(Item.price AS MONEY) AS VARCHAR) AS ValorVenda,
	'R$' + CAST(CAST(Item.freight_value AS MONEY) AS VARCHAR) AS FretePago,
	Prod.category_name AS CategoriaProduto,
	Sell.seller_city + '-' + Sell.seller_state AS RegiaoVendedor,
	Cust.customer_city + '-' + Cust.customer_state AS RegiaoComprador,
	Ord.purchase_timestamp AS DataCompra,
	Pay.payment_type AS FormaPagamento
FROM tb_OlistECommerce_OrdersItems Item
INNER JOIN tb_OlistECommerce_Products Prod
	ON Item.product_id = Prod.product_id
INNER JOIN tb_OlistECommerce_Seller Sell
	ON Item.seller_id = Sell.seller_id
INNER JOIN tb_OlistECommerce_Orders Ord
	ON Item.order_id = Ord.order_id
INNER JOIN tb_OlistECommerce_Customer Cust
	ON Ord.customer_id = Cust.customer_id
INNER JOIN tb_OlistECommerce_OrderPayments Pay
	ON Ord.order_id = Pay.order_id
ORDER BY Item.price DESC