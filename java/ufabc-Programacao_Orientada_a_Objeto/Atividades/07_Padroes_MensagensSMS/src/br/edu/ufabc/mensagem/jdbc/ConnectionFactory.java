package br.edu.ufabc.mensagem.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;

public class ConnectionFactory {

	public Connection getConnection(){
		System.out.println("Conectando ao banco de dados");
		try {
			String url = "jdbc:postgresql://localhost/trocasms";
			return DriverManager.getConnection(url, "postgres", "ufabc");
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

	}

}
