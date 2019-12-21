--Section 3, Módulo 8, Arquivo 2 - PIVOT

/*****************************************************************
							PIVOT

	Antes de demonstrar o comando PIVOT, vamos preparar algumas
queries responsáveis por trazer os dados e consolidar os comandos
de uma forma mais sólida e simples.
*****************************************************************/

--0. Visualizando Tabela Derivada
SELECT P.ProductID, PC.Name, ISNULL(P.Color, 'Uncolored') AS Color
FROM SalesLT.ProductCategory PC
INNER JOIN SalesLT.Product P
ON PC.ProductCategoryID = P.ProductCategoryID

--0.1 Facilitando consulta criando uma VIEW
CREATE VIEW SalesLT.vProductColor
AS
(
	SELECT P.ProductID, PC.Name, ISNULL(P.Color, 'Uncolored') AS Color
	FROM SalesLT.ProductCategory PC
	INNER JOIN SalesLT.Product P
	ON PC.ProductCategoryID = P.ProductCategoryID
);

--0.2 Pivotando dados a partir da coluna Color
SELECT * FROM SalesLT.vProductColor
PIVOT(COUNT(ProductID) FOR Color IN([Red],[Blue],[Black],[Silver],[Yellow],[Grey], [Multi], [Uncolored])) AS PVT
ORDER BY Name;

/*****************************************************************
							UNPIVOT

	O comando UNPIVOT realiza o procedimento contrário ao comando
PIVOT, recebendo então uma tabela já pivotada no formato exemplificado
acima e transformando-a de modo a passar as colunas pivotadas para
as linhas da nova tabela criada. Para exemplificar o UNPIVOT, vamos
antes criar uma tabela temporária para receber os dados pivotados.
*****************************************************************/

--1. Criando tabela temporária
CREATE TABLE #ProductColorPivot
(Name varchar(50), Red int, Blue int, Black int, Silver int, Yellow int, Grey int , multi int, uncolored int);

--1.1 Inserindo dados na tabela temporária
INSERT INTO #ProductColorPivot 
SELECT * FROM SalesLT.vProductColor
PIVOT(COUNT(ProductID) FOR Color IN([Red],[Blue],[Black],[Silver],[Yellow],[Grey], [Multi], [Uncolored])) AS PVT 

--1.2 Verificando tabela pivotada
SELECT * FROM #ProductColorPivot

--1.3 Aplicando UNPIVOT
SELECT Name, Color, ProductCount
FROM
(SELECT Name, [Red],[Blue],[Black],[Silver],[Yellow],[Grey], [Multi], [Uncolored]
FROM #ProductColorPivot) pcp
UNPIVOT(ProductCount FOR Color IN ([Red],[Blue],[Black],[Silver],[Yellow],[Grey], [Multi], [Uncolored])) AS ProductCounts

--2. Outro exemplo com UNPIVOT
CREATE TABLE #SalesByQuarter
(ProductID int,
 Q1 money,
 Q2 money,
 Q3 money,
 Q4 money);

INSERT INTO #SalesByQuarter
VALUES
(1, 19999.00, 21567.00, 23340.00, 25876.00),
(2, 10997.00, 12465.00, 13367.00, 14365.00),
(3, 21900.00, 21999.00, 23376.00, 23676.00);

SELECT * FROM #SalesByQuarter;

SELECT ProductID, Period, Revenue
FROM
(SELECT ProductID,
Q1, Q2, Q3, Q4
FROM #SalesByQuarter) sbq
UNPIVOT
(Revenue FOR Period IN (Q1, Q2, Q3, Q4)
) AS RevenueReport

