package br.com.bb.sistemabancario.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import br.com.bb.sistemabancario.jdbc.ConnectionFactory;
import br.com.bb.sistemabancario.modelo.ContaCorrente;

public class ContaCorrenteDAO {
	
	private Connection connection;
	
	//Método Construtor
	public ContaCorrenteDAO() {
		this.connection = new ConnectionFactory().getConnection();
		
		/*Comentários:
		 * Quando o objeto "dao" do tipo ContaCorrenteDAO for criado, é também criada uma conexão com o BD
		 * através de seu método construtor (chama o método getConnection da classe ConnectionFactory).
		 */
	}
	
	//Métodos para Persistência no Banco de Dados
	
	//--------------------------------- INSERE CONTA CORRENTE -----------------------------------------------
	public void insere(ContaCorrente cc) {
		//Método responsável por inserir dados de Conta Corrente no BD (INSERT INTO)
		String sql = "INSERT INTO contacorrente (numero, agencia, descricao, ativa, variacao)"
				+ "values (?, ?, ?, ?, ?)";
		
		try {
			//Cria uma referência do tipo PreparedStatement e passa a string sql a um método connection
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Seta valores
			stmt.setString(1, cc.getNumero());
			stmt.setString(2, cc.getAgencia());
			stmt.setString(3, cc.getDescricao());
			stmt.setBoolean(4, cc.isAtiva());
			stmt.setInt(5, cc.getVariação());
						
			//Executa query
			stmt.execute();
			
			//Fecha Statement
			stmt.close();
			
			System.out.println("Dados inseridos no Banco de Dados");
			
		} catch (SQLException e) {
			throw new RuntimeException();
		}
	}
	
	//--------------------------------- REMOVE CONTA CORRENTE -----------------------------------------------
	
	public void remove(ContaCorrente cc) {
		//Método para REMOVER Conta Corrente do BD (DELETE FROM)
		String sql = "DELETE FROM contacorrente WHERE id = (?)";
		
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Seta valores
			stmt.setInt(1, cc.getId());
			
			//Executa query
			stmt.execute();
			
			//Fecha Statement
			stmt.close();
			
			System.out.println("Conta Corrente de id " + cc.getId() + " removida do BD.");
		} catch(SQLException e) {
			throw new RuntimeException(e);
		}
	}
	
	//--------------------------------- ALTERA CONTA CORRENTE -----------------------------------------------
	
	public void altera(ContaCorrente cc) {
		//Método para ALTERAR dados da Conta Corrente cc (UPDATE SET)
		
		//Instância de string SQL que receberá a query logo em seguida
		String sql = null;
		Scanner sc = new Scanner(System.in);
		
		//Menu de seleção do atributo a ser alterado
		System.out.println("Qual atributo da CC de id " + cc.getId() + " deseja alterar?");
		System.out.println("[ 1 ] para Número\n"
				+ "[ 2 ] pra Agência\n"
				+ "[ 3 ] para Descrição\n"
				+ "[ 4 ] para desativar CC\n"
				+ "[ 5 ] para Variação");
		int set = sc.nextInt();
		
		//Switch Case para determinar o atributo a ser modificado e alterar a query de acordo com a seleção
		switch(set) {
		case 1:
			sql = "UPDATE contacorrente SET numero = ? where id = ?";				
			break;
		case 2:
			sql = "UPDATE contacorrente SET agencia = ? where id = ?";
			break;
		case 3:
			sql = "UPDATE contacorrente SET descricao = ? where id = ?";
			break;
		case 4:
			sql = "UPDATE contacorrente SET ativa = ? where id = ?";
			break;
		case 5:
			sql = "UPDATE contacorrente SET variacao = ? where id = ?";
			break;
		default: System.out.println("Entrada inválida.");
		}
								
		//Lê do usuário o novo Atributo a ser colocado na CC
		System.out.println("\nInsira o novo valor do atributo: ");
		Scanner sc1 = new Scanner(System.in);
		String valor = sc1.nextLine();
		
		//Início das tratativas para persistência
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Se o atributo alterado for "ativa", automticamente seta como "falso" e desativa a CC
			if (set == 4) {
				stmt.setBoolean(1, false);
			} else if (set == 5){ //Se for "variação", transforma o valor em Inteiro
				Integer setVar = Integer.parseInt(valor);
				stmt.setInt(1, setVar);
			} else { //Caso contrário, envia o valor que for digitado pelo usuário.
				stmt.setString(1, valor);
			}		
			
			//Por fim, seta o Id na string SQL
			stmt.setInt(2, cc.getId());
			
			//Executa query e fecha statement.
			stmt.execute();
			stmt.close();
			
			System.out.println("Dados alterados com sucesso.");
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}
	
	//--------------------------------- SELECIONA TODAS AS CONTAS -----------------------------------------------
	
	public List<ContaCorrente> getListaContas(){
		//Método para retornar uma lista com todas as Contas Correntes presentes no BD
		
		//Cria um lista do tipo ContaCorrente e define a string sql de query
		List<ContaCorrente> ccs = new ArrayList<ContaCorrente>();
		String sql = "SELECT * FROM contacorrente ORDER BY id";
		
		try {
			PreparedStatement stmt = connection.prepareStatement(sql);
			
			//Cria um objeto do tipo ResultSet que recebe o método executeQuery() do statement.
			ResultSet rs = stmt.executeQuery(); 
			while (rs.next()) { //Enquanto há Contas a serem lidas...
				//Cria diversos objetos do tipo ContaCorrente e define os parâmetros pelos métodos do ResultSet
				ContaCorrente cc = new ContaCorrente(rs.getString("numero"), rs.getString("agencia"),
						rs.getString("descricao"), rs.getInt("variacao"));
				cc.setId(rs.getInt("id"));
				ccs.add(cc); //Adiciona cada CC criada à lista de ContasCorrentes ccs
			}
			//Ao fim do laço, fecha ResultSet e fecha Statement
			rs.close();
			stmt.close();
			
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
		return ccs; //Retorna uma lista de Contas Correntes que deverá ser lida no método Main
		
		
	}
	
}
