--Section 4, Módulo 10, Arquivo 1 - Variávies

/*****************************************************************
							DECLARE

	O comando DECLARE é utilizado para a criação de variáveis no
SQL Server. Existem diferentes formas de atribuir um valor a uma
variável porém, em todo caso, a sintaxe de declaração é dada por:

				DECLARE @variable_name DATATYPE
*****************************************************************/

--1. Declarando uma variável
DECLARE @City VARCHAR(20) = 'Toronto'
--SET @City = 'Bellevue'

--GO; As variáveis são definidas dentro do escpo da batch e, portanto, com o comando GO, a query abaixo não reconhecerá a variável criada acima

SELECT c.FirstName + ' ' + c.LastName AS Name, a.AddressLine1 AS Address, a.City
FROM SalesLT.Customer c
INNER JOIN SalesLT.CustomerAddress ca
ON c.CustomerID = ca.CustomerID
INNER JOIN SalesLT.Address a
ON ca.AddressID = a.AddressID
WHERE a.City = @City

--2. Atribuindo valores à variável através de um SELECT
DECLARE @Result MONEY
SELECT @Result = MAX(TotalDue)
FROM SalesLT.SalesOrderHeader

PRINT @Result