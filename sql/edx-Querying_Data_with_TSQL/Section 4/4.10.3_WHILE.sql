--Section 4, Módulo 10, Arquivo 3 - WHILE

/*****************************************************************
					WHILE - BEGIN - END

	Os loopings em SQL geralmente não são tão aplicáveis, visto que
estamos interessados em operações baseadas em "sets" e não linha
a linha. Comandos que rodam em batch de uma única vez são mais
eficientes do que programas que iteram através de instâncias

				WHILE <condition>
				BEGIN
					<comandos>
				END
*****************************************************************/

--0. Criando tabela de exemplo

IF OBJECT_ID('SalesLT.DemoTable') IS NOT NULL
	BEGIN
	DROP TABLE SalesLT.DemoTable
	END
GO


CREATE TABLE SalesLT.DemoTable
(ID INT IDENTITY(1,1),
Description Varchar(20),
CONSTRAINT [PK_DemoTable] PRIMARY KEY CLUSTERED(ID) 
)
GO

--1. Exemplo WHILE
DECLARE @Counter INT = 1

WHILE @Counter <= 5
BEGIN
	INSERT SalesLT.DemoTable(Description)
	VALUES ('ROW ' + CONVERT(VARCHAR(5), @Counter))
	SET @Counter = @Counter + 1
END

SELECT * FROM SalesLT.DemoTable