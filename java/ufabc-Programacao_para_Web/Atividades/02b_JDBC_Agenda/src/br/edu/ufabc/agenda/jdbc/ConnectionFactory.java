package br.edu.ufabc.agenda.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;

/*A classe ConnectionFactory é uma Fábrica de Conexões e tem como responsabilidade criar as conexões com o BD.
 * 
 * Dessa forma, há uma facilidade na manutenção: quando precisar alterar a forma para obter conexões (ou mesmo
 * a mudança do driver do Banco de Dados), o padrão Factory faz o encapsulamento da construçãod e objetos.	
 */

public class ConnectionFactory {
	
	public Connection getConnection() {
		//O método getConnection() retorna um objeto da classe Connection
		try {
			String url = "jdbc:postgresql://localhost/agendajdbc"; //String para conexão com o Banco
			System.out.println("Conexão com o BD realizada com sucesso.\n");
			return DriverManager.getConnection(url, "thipanini", "coyote94"); //Retorno do tipo Connection
			//Parâmetros DriverManager.getConnection: url, usuário (owner), senha
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

}
