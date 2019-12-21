--Section 3, Módulo 6, Arquivo 3 - CROSS APPLY

--1. Retornar a coluna OrderID e o valor máximo do preço unitário (UnitPrice)
SELECT SOH.SalesOrderID, MUP.MaxUnitPrice
FROM SalesLT.SalesOrderHeader SOH
CROSS APPLY SalesLT.udfMaxUnitPrice(SOH.SalesOrderID) AS MUP
ORDER BY SOH.SalesOrderID