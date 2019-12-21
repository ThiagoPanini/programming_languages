--Section 2, Módulo 5, Arquivo 2 - LOGICAL FUNCTIONS

--1. ISNUMERIC: Retornando consulta apenas onde o Tamanho do Produto é do tipo numérico
SELECT Name, Size FROM SalesLT.Product
WHERE ISNUMERIC(Size) = 1

--2. IIF e ISNUMERIC: Verificando se um tipo primitivo é ou não numérico
SELECT 
	Name AS NomeProduto, 
	Size AS TamanhoProduto,
	IIF(ISNUMERIC(Size)=1, 'Numeric', 'Non-Numeric') AS TipoTamanho
FROM SalesLT.Product