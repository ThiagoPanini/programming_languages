--Section 4, Módulo 11, Arquivo 2 - TRY... CATCH

/*****************************************************************
							TRY CATCH

	Sintaxe:

	BEGIN TRY:
		<sql_statement>
	END TRY
	BEGIN CATCH
		<sql_statement>
	END CATCH

Informações do erro que podem ser coletadas no bloco CATCH:
- @@ERROR
- ERROR_NUMBER()
- ERROR_MESSAGe()
- ERROR_SEVERITY()
- ERROR_STATE()
- ERROR_PROCEDURE()
- ERROR_LINE()
*****************************************************************/

--1. Tratando erros utilizando TRY CATCH
BEGIN TRY
	UPDATE SalesLT.Product
	SET ProductNumber = ProductID / ISNULL(Weight, 0);
END TRY
BEGIN CATCH
	PRINT 'The following error occurred:'
	PRINT ERROR_MESSAGE();
END CATCH;

--2. Tratando erros com TRY CATCH e relançando erro com o THROW
BEGIN TRY
	UPDATE SalesLT.Product
	SET ProductNumber = ProductID / ISNULL(Weight, 0);
END TRY
BEGIN CATCH
	PRINT 'The following error occurred:'
	PRINT ERROR_MESSAGE();
	THROW;
END CATCH;

--3. Trabalhando com procedures responsáveis por armazenar logs de erros
BEGIN TRY
	UPDATE SalesLT.Product
	SET ProductNumber = ProductID / ISNULL(Weight, 0);
END TRY
BEGIN CATCH
	DECLARE @ErrorLogID AS INT, @ErrorMsg AS VARCHAR(250);
	EXECUTE dbo.uspLogError @ErrorLogID OUTPUT;
	SET @ErrorMsg = 'The update failed because of an error. View error #'
					+ CAST(@ErrorLogID AS VARCHAR)
					+ ' in the error log for details.';
	THROW 50001, @ErrorMsg, 0;
END CATCH

--3.1 Visualizando Log
SELECT * FROM dbo.ErrorLog;