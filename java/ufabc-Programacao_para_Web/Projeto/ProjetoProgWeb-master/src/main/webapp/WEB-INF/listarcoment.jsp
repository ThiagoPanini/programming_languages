<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
	pageEncoding="ISO-8859-1"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form"%>

<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Lista de Comentários em Notícias</title>
</head>
<body>

	<table border="1">

		<thead>
			<tr>
				<th><b>Nome</b></th>
				<th><b>Data</b></th>
				<th><b>Mensagem</b></th>
			</tr>
		</thead>
		<tbody>
			<c:forEach items="${comentarios}" var="c">
				<tr>
					<td><c:out value="${c.nome}"></c:out></td>
					<td><c:out value="${c.data}"></c:out></td>
					<td><c:out value="${c.mensagem}"></c:out></td>

					<td><a href="/edit/${c.id_comentario}">
							<button type="submit">Editar</button>
					</a></td>
				</tr>
			</c:forEach>
		</tbody>
	</table>	
	<br>
	${msg}
	<br><br>
	<a href="index">Ver Notícias</a>
	<br>
</body>
</html>
