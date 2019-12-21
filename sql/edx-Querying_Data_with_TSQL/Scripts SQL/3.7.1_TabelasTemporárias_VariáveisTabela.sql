--Section 3, Módulo 7, Arquivo 2 - Tabelas Temporárias, Variáveis TABLE

--1. Criando tabela temporária
CREATE TABLE #tmpProducts
(ProductID INTEGER,
ProductName VARCHAR(5));

--1.1 Selecionando dados da tabela temporária
SELECT * FROM #tmpProducts;

--2. Criando variáveis do tipo TABLE
DECLARE @varProducts TABLE
(ProductID INTEGER,
ProductName VARCHAR(50));

--2.1 O escopo da variável é o batch do comando
SELECT * FROM @varProducts;
--Obs: SELECT e DECLARE devem ser executados juntos
--Não é possível executar o SELECT individualmente

--------------------------------------------------------------

--3. Tabelas Temporárias
CREATE TABLE #Colors
(Color VARCHAR(15));

INSERT INTO #Colors
SELECT DISTINCT Color FROM SalesLT.Product

SELECT * FROM #Colors

--4. Variáveis do tipo Tabela
DECLARE @Colors AS TABLE
(Color VARCHAR(15));

INSERT INTO @Colors
SELECT DISTINCT Color FROM SalesLT.Product

--A declaração, inserção e seleção devem ser executadas em uma mesma batch
SELECT * FROM @Colors