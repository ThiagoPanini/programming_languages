package br.edu.ufabc.mensagem.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Scanner;

import br.edu.ufabc.mensagem.jdbc.ConnectionFactory;
import br.edu.ufabc.mensagem.modelo.Mensagem;

public class MensagemDAO {
	
	private Connection connection;
	
	public MensagemDAO() {
		this.connection = new ConnectionFactory().getConnection();
	}
	
	//Método para INSERIR Mensagens no BD
	public void insereMensagem(Mensagem msg) {
		String sql = "INSERT INTO mensagem (celorigem, celdestino, texto) values (?, ?, ?)";
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			stmt.setString(1, msg.getCelOrigem());
			stmt.setString(2, msg.getCelDestino());
			stmt.setString(3, msg.getTexto());			
			stmt.execute();
			stmt.close();
		} catch (SQLException e){
			throw new RuntimeException(e);
		}
	}
	
	public void deletaMensagem(Mensagem msg) {
		String sql, cel;
		Integer opt2;
		
		System.out.println("\n- - - DELETANDO MENSAGEM - - -");
		System.out.println("Deseja deletar por:"
				+ "\n[ 1 ] ID"
				+ "\n[ 2 ] Celular de Origem"
				+ "\n[ 3 ] Celular de Destino");
		Scanner sc = new Scanner(System.in);
		Integer opt = Integer.parseInt(sc.nextLine());
		
		switch(opt){
		case 1:
			sql = "DELETE FROM mensagem WHERE id = ?";
			System.out.println("\nInsira o ID da mensagem a ser deletada: ");
			opt2 = Integer.parseInt(sc.nextLine());
			try {
				PreparedStatement stmt = connection.prepareStatement(sql);
				stmt.setInt(1, opt2);
				stmt.execute();
				stmt.close();	
				System.out.println("Mensagem removida com sucesso!");
			} catch (SQLException e){
				throw new RuntimeException(e);
			}			
			break;
		case 2:
			sql = "DELETE FROM mensagem WHERE celorigem = ?";
			System.out.println("\nInsira o Celular de Origem cujas mensagens ser�o deletadas: ");
			cel = sc.nextLine();
			try {
				PreparedStatement stmt = connection.prepareStatement(sql);
				stmt.setString(1, cel);
				stmt.execute();
				stmt.close();	
				System.out.println("Mensagem removida com sucesso!");
			} catch (SQLException e){
				throw new RuntimeException(e);
			}	
			break;
		case 3:
			sql = "DELETE FROM mensagem WHERE celdestino = ?";
			System.out.println("\nInsira o Celular de Destino cujas mensagens ser�o deletadas: ");
			cel = sc.nextLine();
			try {
				PreparedStatement stmt = connection.prepareStatement(sql);
				stmt.setString(1, cel);
				stmt.execute();
				stmt.close();	
				System.out.println("Mensagem removida com sucesso!");
			} catch (SQLException e){
				throw new RuntimeException(e);
			}	
			break;
		default: sql = "DELETE FROM mensagem WHERE id = ?";
		System.out.println("\nInsira o ID da mensagem a ser deletada: ");
		opt2 = Integer.parseInt(sc.nextLine());
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			stmt.setInt(1, opt2);
			stmt.execute();
			stmt.close();		
			System.out.println("Mensagem removida com sucesso!");
		} catch (SQLException e){
			throw new RuntimeException(e);
		}	
		}
	}
}
	
		