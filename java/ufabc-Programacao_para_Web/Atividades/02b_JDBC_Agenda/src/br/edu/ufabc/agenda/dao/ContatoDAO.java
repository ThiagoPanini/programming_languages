package br.edu.ufabc.agenda.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import br.edu.ufabc.agenda.jdbc.ConnectionFactory;
import br.edu.ufabc.agenda.modelo.Contato;

public class ContatoDAO {
	
	private Connection connection;
	
	//Método Construtor
	public ContatoDAO() {
		this.connection = new ConnectionFactory().getConnection();
		
		/*Comentários:
		 * Quando o objeto "dao" do tipo ContaCorrenteDAO for criado, é também criada uma conexão com o BD
		 * através de seu método construtor (chama o método getConnection da classe ConnectionFactory).
		 */
	}
	
	//Métodos para Persistência no Banco de Dados
	
	//--------------------------------- INSERE CONTATO -----------------------------------------------
	public void insere(Contato ct) {
		//Método responsável por inserir dados de Conta Corrente no BD (INSERT INTO)
		String sql = "INSERT INTO contato (nome, email, endereco) values (?, ?, ?)";
		
		try {
			//Cria uma referência do tipo PreparedStatement e passa a string sql a um método connection
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Seta valores
			stmt.setString(1, ct.getNome()); 
			stmt.setString(2, ct.getEmail());
			stmt.setString(3, ct.getEndereço());
						
			//Executa query
			stmt.execute();
			
			//Fecha Statement
			stmt.close();
			
			System.out.println("Dados inseridos no Banco de Dados");
			
		} catch (SQLException e) {
			throw new RuntimeException();
		}
	}
	
	//--------------------------------- REMOVE CONTATO -----------------------------------------------
	
	public void remove(Contato ct) {
		//Método para REMOVER um item do BD (DELETE FROM)
		String sql = "DELETE FROM contato WHERE id = (?)";
		
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Seta valores
			stmt.setInt(1, ct.getId());
			
			//Executa query
			stmt.execute();
			
			//Fecha Statement
			stmt.close();
			
			System.out.println("Contato de id " + ct.getId() + " removido do BD.");
		} catch(SQLException e) {
			throw new RuntimeException(e);
		}
	}
	
	//--------------------------------- ALTERA CONTA CORRENTE -----------------------------------------------
	
	public void altera(Contato ct) {
		//Método para ALTERAR dados da tabela no BD (UPDATE SET)
		
		//Instância de string SQL que receberá a query logo em seguida
		String sql = null;
		Scanner sc = new Scanner(System.in);
		
		//Menu de seleção do atributo a ser alterado
		System.out.println("Qual atributo do Contato de id " + ct.getId() + " deseja alterar?");
		System.out.println("[ 1 ] para Nome\n"
				+ "[ 2 ] pra E-mail\n"
				+ "[ 3 ] para Endereço");
		int set = Integer.parseInt(sc.nextLine());
		
		//Switch Case para determinar o atributo a ser modificado e alterar a query de acordo com a seleção
		switch(set) {
		case 1:
			sql = "UPDATE contato SET nome = ? where id = ?";				
			break;
		case 2:
			sql = "UPDATE contato SET email = ? where id = ?";
			break;
		case 3:
			sql = "UPDATE contato SET endereco = ? where id = ?";
			break;
		default: System.out.println("Entrada inválida.");
		}
								
		//Lê do usuário o novo Atributo a ser colocado no Contato
		System.out.println("\nInsira o novo valor do atributo: ");
		String valor = sc.nextLine();
		
		//Início das tratativas para persistência
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			stmt.setString(1, valor);	
			stmt.setInt(2, ct.getId());
				
			//Executa query e fecha statement.
			stmt.execute();
			stmt.close();
			
			System.out.println("Dados alterados com sucesso.");
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}
	
	//--------------------------------- SELECIONA TODOS OS CONTATOS -----------------------------------------------
	
	public List<Contato> getListaContatos(){
		//Método para retornar uma lista com todos os itens presentes no BD
		
		//Cria um lista do tipo da classe a ser retornada e define a string sql de query
		List<Contato> cts = new ArrayList<Contato>();
		String sql = "SELECT * FROM contato ORDER BY id";
		
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Cria um objeto do tipo ResultSet que recebe o método executeQuery() do statement.
			ResultSet rs = stmt.executeQuery(); 
			while (rs.next()) { //Enquanto há itens a serem lidos...
				//Cria diversos objetos do tipo ContaCorrente e define os parâmetros pelos métodos do ResultSet
				Contato ct = new Contato(rs.getString("nome"), rs.getString("email"),
						rs.getString("endereco"));
				ct.setId(rs.getInt("id"));
				cts.add(ct); //Adiciona cada item criado à lista geral de itens
			}
			//Ao fim do laço, fecha ResultSet e fecha Statement
			rs.close();
			stmt.close();
			
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
		return cts; //Retorna uma lista de itens que deverá ser lida no método Main
		
		
	}
	
}
