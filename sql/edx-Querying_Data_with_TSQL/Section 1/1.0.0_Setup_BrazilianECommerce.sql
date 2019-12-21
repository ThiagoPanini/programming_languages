-- Tabela com dados sobre o cliente
CREATE TABLE tb_OlistECommerce_Customer(
	customer_id VARCHAR(255) PRIMARY KEY,
	customer_unique_id VARCHAR(255),
	zip_code_prefix VARCHAR(255),
	customer_city VARCHAR(255),
	customer_state VARCHAR(255)
);

--Tabela com dados sobre o vendedor
CREATE TABLE tb_OlistECommerce_Seller (
	seller_id VARCHAR(255) PRIMARY KEY,
	zip_code_prefix VARCHAR(255),
	seller_city VARCHAR(255),
	seller_state VARCHAR(255)
);

--Ajustando alguns dados incoerentes
UPDATE tb_OlistECommerce_Seller SET seller_state = 'RJ' WHERE seller_city = '"rio de janeiro'
UPDATE tb_OlistECommerce_Seller SET seller_state = 'RS' WHERE seller_city = '"rio de janeiro'
UPDATE tb_OlistECommerce_Seller SET seller_city = 'rio de janeiro' where seller_city = '"04482255"'
UPDATE tb_OlistECommerce_Seller SET seller_city = SUBSTRING(seller_city, 2, len(seller_city)) WHERE seller_city like '"%'

--Tabela com dados sobre os pedidos (tabela central)
CREATE TABLE tb_OlistECommerce_Orders (
	order_id VARCHAR(255) PRIMARY KEY,
	customer_id VARCHAR(255) NOT NULL FOREIGN KEY REFERENCES tb_OlistECommerce_Customer(customer_id),
	order_status VARCHAR(255),
	purchase_timestamp DATETIME,
	approved_at DATETIME,
	delivered_carrier_date DATETIME,
	delivered_customer_date DATETIME,
	estimated_delivered_date DATE
);

--Tabela com dados sobre produtos
CREATE TABLE tb_OlistECommerce_Products (
	product_id VARCHAR(255) PRIMARY KEY,
	category_name VARCHAR(255),
	name_length DECIMAL(9, 1),
	description_length DECIMAL(9, 1),
	photos_qty DECIMAL(9, 1),
	prod_weight_g DECIMAL(9, 1),
	prod_length_cm DECIMAL(9, 1),
	prod_height_cm DECIMAL(9, 1),
	prod_width_cm DECIMAL(9, 1)
);

--Tabela com dados de pedidos e itens
CREATE TABLE tb_OlistECommerce_OrdersItems (
	order_id VARCHAR(255) FOREIGN KEY REFERENCES tb_OlistECommerce_Orders(order_id),
	order_item_it TINYINT,
	product_id VARCHAR(255) FOREIGN KEY REFERENCES tb_OlistECommerce_Products(product_id),
	seller_id VARCHAR(255) FOREIGN KEY REFERENCES tb_OlistECommerce_Seller(seller_id),
	shipping_limit_date DATETIME,
	price FLOAT,
	freight_value FLOAT
);
--Adaptação para corrigir erro de importação nas colunas "price" e "freight_value"
UPDATE tb_OlistECommerce_OrdersItems SET price = price/100;
UPDATE tb_OlistECommerce_OrdersItems SET freight_value = freight_value/100

--Criando tabela para pagamentos
CREATE TABLE tb_OlistECommerce_OrderPayments (
	order_id VARCHAR(255) FOREIGN KEY REFERENCES tb_OlistECommerce_Orders(order_id),
	payment_sequential TINYINT,
	payment_type VARCHAR(255),
	payment_installments TINYINT,
	payment_value FLOAT,
);
--Adaptação para corrigir erro de importação nas colunas "payament_value"
UPDATE tb_OlistECommerce_OrderPayments SET payment_value = payment_value/100;

-- Tabela com dados geográficos de latitude e longitude
CREATE TABLE tb_OlistECommerce_Geolocation (
	zip_code_prefix VARCHAR(255),
	geolocation_lat DECIMAL(9, 6),
	geolocation_lng DECIMAL(9, 6),
	geolocation_city VARCHAR(255),
	geolocation_state VARCHAR(255)
);