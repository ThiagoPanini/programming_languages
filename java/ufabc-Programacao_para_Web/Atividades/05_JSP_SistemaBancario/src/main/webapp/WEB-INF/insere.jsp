<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
	pageEncoding="ISO-8859-1"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form"%>
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Salvando Conta Corrente</title>
</head>
<body>

	<form id="contacorrente" action="/save" method="POST">
		<label>Número: <input name="numero" type="text" required>
		</label>
		<br /> <label>Agência: <input name="agencia" type="text"
			required>
		</label>
		<br /> <label>Descrição: <textarea rows="2" cols="30"
				name="descricao"> Descrição da conta </textarea>
		</label>
		<br /> <label>Situação: <input type="radio" name="situacao"
			value="true" /> Ativa  <input type="radio" name="situacao"
			value="false" /> Inativa <br />
		</label>
		<br /> <label>Variação: <input name="variacao" type="number"
			required min="0" max="20" step="1">
		</label><br>
		<br /> <input type="submit" name="action" value="Salvar conta" />
	</form>
	${msg}
	<br>
	<a href="index">Listar Contas Corrente</a>
	<br>
</body>
</html>