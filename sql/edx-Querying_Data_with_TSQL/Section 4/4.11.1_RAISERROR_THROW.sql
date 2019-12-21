--Section 4, Módulo 11, Arquivo 1 - RAISEERROR e THROW

/*****************************************************************
							Erros

	Os erros em T-SQL são lançados seguinto a seguinte estrutura:

- Error number: número único que identifica um erro em específico;
- Error message: mensagem descrevendo o erro;
- Severity: indicação numérica da seriedade (1 a 25);
- State: código de estado interno para a condição do database
- Procedure: nome da SP u Trigger onde o erro ocorreu;
- Line number: em qual statemente o erro ocorreu
*****************************************************************/

/*****************************************************************
					  RAISERROR

	Lança um erro específico na tablea sys.messages. Sintaxe:

	RAISEERROR('Error message', 16 <severity>, 0 <state>)
*****************************************************************/

/*****************************************************************
					  THROW

	Também lança um erro, porém de uma maneira ligeiramente diferente

	THROE 50001, 'Error message', 0

	(A severidade é sempre 16 para o lançamento de erros com o THROW)
*****************************************************************/

--1. Atualizando tabela e lançando erros (RAISEERROR)
INSERT INTO SalesLT.SalesOrderDetail (SalesOrderID, OrderQty, ProductID, UnitPrice, UnitPriceDiscount)
VALUES (1000000, 1, 680, 1431.50, 0.00)

UPDATE SalesLT.Product SET DiscontinuedDate = GETDATE()
WHERE ProductID = 0;

IF @@ROWCOUNT < 1
	RAISERROR('The product was not found - no products have been updated', 16, 0);

--2. Atualizando tabela e lançando erros (THROW)
UPDATE SalesLT.Product SET DiscontinuedDate = GETDATE()
WHERE ProductID = 0;

IF @@ROWCOUNT < 1
	THROW 50001, 'The product was not found - no products have been updated', 0;