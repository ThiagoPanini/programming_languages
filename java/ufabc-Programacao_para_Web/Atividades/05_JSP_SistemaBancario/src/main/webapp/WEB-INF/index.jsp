<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
	pageEncoding="ISO-8859-1"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form"%>

<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Lista de Contas Corrente</title>
</head>
<body>

	<table border="1">

		<thead>
			<tr>
				<th><b>Número</b></th>
				<th><b>Agencia</b></th>
				<th><b>Descricao</b></th>
				<th><b>Ação</b></th>
			</tr>
		</thead>
		<tbody>
			<c:forEach items="${ccs}" var="cc">
				<tr>
					<td><c:out value="${cc.numero}"></c:out></td>
					<td><c:out value="${cc.agencia}"></c:out></td>
					<td><c:out value="${cc.descricao}"></c:out></td>

					<td><a href="/edit/${cc.id}">
							<button type="submit">Editar</button>
					</a></td>
				</tr>
			</c:forEach>
		</tbody>
	</table>	
	<br>
	${msg}
	<br><br>
	<a href="insere">Adicionar Conta Corrente</a>
	<br>
</body>
</html>
