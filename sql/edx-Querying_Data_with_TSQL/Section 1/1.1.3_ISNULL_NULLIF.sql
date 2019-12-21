--Section 1, Módulo 1, Arquivo 3 - Trabalhando com dados NULOS

--1. Troque os valores nulos da conversão da coluna Size para Integer da tabela Product com TRY_CAST por 0
SELECT ProductNumber, ISNULL(TRY_CAST(Size AS Integer), 0) AS NumericSize FROM SalesLT.Product;

--2. Retorne as colunas ProductNumber, Concatenação entre Color e Size com "allias" ProductDetails
SELECT ProductNumber, Color, Size FROM SalesLT.Product;
SELECT ProductNumber, Color, Size, (Color + ', ' + Size) AS ProductDetails FROM SalesLT.Product;
SELECT 
	ProductNumber, Color, Size, 
	(Color + ', ' + Size) AS ProductDetails,
	ISNULL(Color, '') + ', ' + ISNULL(Size, '') AS ProductDetails2
FROM SalesLT.Product

--3. Retornar ProductNumber e Color, porém com valor NULO em Color se Color='Multi'
SELECT ProductNumber, Color, NULLIF(Color, 'Multi') AS NonMultiColor FROM SalesLT.Product;