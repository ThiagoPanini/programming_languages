--Section 3, Módulo 8, Arquivo 1 - GROUPING SETS

/*******************************************************
	SELECT <column list with aggregate(s)>
	FROM <source>
	GROUP BY
	GROUPING SETS
	(
		<column_name>, --one or more columns
		<column_name>, --one or more columns
		() --empty parenthesis if aggregating all rows
	);
*******************************************************/

-- GROUPING SETS permitem múltiplos agrupamentos
SELECT cat.ParentProductCategoryName, cat.ProductCategoryName, COUNT(prd.ProductID) AS Products, 
GROUPING_ID(cat.ParentProductCategoryName) AS GID_Cat1, GROUPING_ID(cat.ProductCategoryName) AS GID_Cat2
FROM SalesLT.vGetAllCategories cat
LEFT JOIN SalesLT.Product prd
ON cat.ProductCategoryID = prd.ProductCategoryID
--GROUP BY cat.ParentProductCategoryName, cat.ProductCategoryName
--GROUP BY GROUPING SETS (cat.ParentProductCategoryName, cat.ProductCategoryName, ())
GROUP BY ROLLUP (cat.ParentProductCategoryName, cat.ProductCategoryName)
--GROUP BY CUBE (cat.ParentProductCategoryName, cat.ProductCategoryName)
ORDER BY cat.ParentProductCategoryName, cat.ProductCategoryName;

/*******************************************************
					GROUP BY
	Agrupamento clássico a partir das colunas agregadas 
e utilizadas no comando de seleção. No exemplo acima, 
são contados a quantidade de Produtos por Categoria 
Principal e Categoria Específica
*******************************************************/

/*******************************************************
				  GROUPING SETS
	Agrupamento que considera, de forma separada, as 
colunas agregadas. No exemplo acima, temos:

	- Uma primeira linha contendo <NULL> para ambas as 
colunas, indicando a quantidade total do agrupamento
	- Um agrupamento pela segunda categoria, indicando
a quantidade total agrupada para esta coluna
	- Um agrupamento pela primeira categoria, indicando
a quantidade total agrupada para esta coluna
*******************************************************/

/*******************************************************
				  GROUP BY ROLLUP
	Nesse agrupamento, temos uma primeira linha com a 
quantidade total agrupada em ambas as categorias. Na
sequência, antes de cada detalhadamento agrupado 
(semelhante ao GROUP BY), temos uma linha que indica
a quantidade agrupada da coluna-mãe (primeira categoria)
*******************************************************/

/*******************************************************
				  GROUP BY ROLLUP
	Nesse agrupamento, ainda temos uma primeira linha com
o total geral, seguido de um agrupamento pela segunda
categoria. Após, cada categoria pai é agrupada, retornando
seu respectivo total seguido de um agrupamento de ambas
as categorias
*******************************************************/

/*******************************************************
				    APRIMORANDO
	
	Para deixar o agrupamento mais sofisticado, vamos
inserir uma coluna de Level, informando ao usuário um
indicativo de subtotal para cada grupo
*******************************************************/

SELECT 
	cat.ParentProductCategoryName AS CategoriaMae, 
	cat.ProductCategoryName AS SubCategoria, 
	IIF(GROUPING_ID(cat.ParentProductCategoryName) = 1 AND GROUPING_ID(cat.ProductCategoryName) = 1, 'Total', --Condição 1
		IIF(GROUPING_ID(cat.ProductCategoryName) = 1, cat.ParentProductCategoryName + ' Subtotal', --ELSE IF
			cat.ProductCategoryName + ' Subtotal')) AS Level, --ELSE
	COUNT(prd.ProductID) AS Products
FROM SalesLT.vGetAllCategories cat
LEFT JOIN SalesLT.Product prd
ON cat.ProductCategoryID = prd.ProductCategoryID
--GROUP BY cat.ParentProductCategoryName, cat.ProductCategoryName
--GROUP BY GROUPING SETS (cat.ParentProductCategoryName, cat.ProductCategoryName, ())
GROUP BY ROLLUP (cat.ParentProductCategoryName, cat.ProductCategoryName)
--GROUP BY CUBE (cat.ParentProductCategoryName, cat.ProductCategoryName)
ORDER BY cat.ParentProductCategoryName, cat.ProductCategoryName;


IIF(GROUPING_ID(a.CountryRegion) = 1 AND GROUPING_ID(a.StateProvince) = 1, 
'Total', IIF(GROUPING_ID(a.StateProvince) = 1, a.CountryRegion + ' Subtotal', a.StateProvince + ' Subtotal')) AS Level,
