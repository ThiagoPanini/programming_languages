--Section 3, Módulo 7, Arquivo 3 - Table-Valued Function

--1. Criando uma função que retorna uma tabela virtual
CREATE FUNCTION SalesLT.udfCustomersByCity (@City AS VARCHAR(20))
RETURNS TABLE
AS
RETURN 
(
	SELECT c.CustomerID, FirstName, LastName, AddressLine1, City, StateProvince
	FROM SalesLT.Customer c 
	INNER JOIN SalesLT.CustomerAddress ca
	ON c.CustomerID = ca.CustomerID
	INNER JOIN SalesLT.Address a
	ON ca.AddressID = a.AddressID
	WHERE City = @City
)

--1.1 Executando função e obtendo uma tabela como retorno
SELECT * FROM SalesLT.udfCustomersByCity('Bellevue')