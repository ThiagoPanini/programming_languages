SELECT TOP 100 * FROM tb_OlistECommerce_Customer;
SELECT TOP 100 * FROM tb_OlistECommerce_Geolocation;
SELECT TOP 100 * FROM tb_OlistECommerce_OrderPayments;
SELECT TOP 100 * FROM tb_OlistECommerce_Orders;
SELECT TOP 100 * FROM tb_OlistECommerce_OrdersItems;
SELECT TOP 100 * FROM tb_OlistECommerce_Products;
SELECT TOP 100 * FROM tb_OlistECommerce_Seller;

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



select 