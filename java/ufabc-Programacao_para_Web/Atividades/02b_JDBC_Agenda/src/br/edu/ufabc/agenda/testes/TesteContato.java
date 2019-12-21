package br.edu.ufabc.agenda.testes;

import java.sql.SQLException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import br.edu.ufabc.agenda.dao.ContatoDAO;
import br.edu.ufabc.agenda.modelo.Contato;

public class TesteContato {

	public static void main(String[] args) {
		
		String nome, email, end;
		Scanner sc = new Scanner(System.in);
		
		ContatoDAO dao = new ContatoDAO();
		
		//Menu para seleção do usuário
		System.out.println("- - - SISTEMA DE GERENCIAMENTO DE CONTATOS - - -\n"
				+ "Escolha uma opção: \n"
				+ "[ 1 ] para inserir um contato na agenda (Fornecer parâmetros)\n"
				+ "[ 2 ] para inserir um contato aleatório (Parâmetros aleatórios)\n"
				+ "[ 3 ] para remover um contato\n"
				+ "[ 4 ] para alterar um contato\n"
				+ "[ 5 ] para visualizar a listagem de contatos na agenda");		
		Integer opt = Integer.parseInt(sc.nextLine());
		
		//Toma ação baseada no valor inserido pelo usuário
		switch(opt) {
		case 1:
			//Solicita dados para criação da CC
			System.out.println("- - - CADASTRO DE CONTATO - - -");
			System.out.print("Nome: ");
			nome = sc.nextLine();
			System.out.print("E-mail: ");
			email = sc.nextLine();
			System.out.print("Endereço: ");
			end = sc.nextLine();
			
			Contato ct = new Contato(nome, email, end);
	
			try {
				dao.insere(ct);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 2:
			//Cria dados aleatórios para inserção de CC no BD
			Random rd = new Random();
			
			System.out.println("- - - GERANDO CONTATO ALEATÓRIO (TESTE) - - -");
		
			String [] nomes = {"Thiago Henrique", "Jakeline G.", "Dorinha P.", "Antonio Carlos", "Ximê",
					"Gracilda", "Pikachu", "Mew", "Goku", "Axl Rose"};
			
			nome = nomes[rd.nextInt(nomes.length)];
			email = "teste@teste.com.br";
			end = "Rua " + rd.nextInt(15);
			
			Contato ct1 = new Contato(nome, email, end);

			try {
				dao.insere(ct1);
				System.out.println(
						  "Nome: " + ct1.getNome()
						+ "\nE-mail: " + ct1.getEmail()
						+ "\nEndereço: " + ct1.getEndereço());
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 3:
			//Solicita inserção do id da CC a ser removida
			System.out.println("- - - DELETANDO CONTATO - - -");
			System.out.println("Insira o id do contato que deseja deletar do Banco: ");
			Integer idremover = Integer.parseInt(sc.nextLine());
			
			try {
				List<Contato> cts = dao.getListaContatos();
				for (Contato contato : cts) {
					if (contato.getId() == idremover) {
						//Verifica se o id inserido é o mesmo de cada CC presente no BD
						dao.remove(contato);
					}
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 4:
			//Solicita id da CC a ser alterada
			System.out.println("- - - ALTERANDO UM CONTATO - - -");
			System.out.println("Insira o id da conta que deseja alterar no Banco: ");
			Integer idalterar = Integer.parseInt(sc.nextLine());
			
			try {
				List<Contato> cts = dao.getListaContatos();
				for (Contato contato : cts) {
					if (contato.getId() == idalterar) {
						//Verifica se há o id inserido dentro das CCs presentes no BD
						dao.altera(contato);
					}
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 5:
			//Imprime o stataus de todos os Contatos no BD
			System.out.println("- - - MOSTRANDO TODOS OS CONTATOS DA AGENDA - - -");
			try {
				List<Contato> cts = dao.getListaContatos();
				for (Contato contato : cts) {
					contato.statusContato();
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
		default: System.out.println("Entrada inválida.");
			
		}
		
	}

}
