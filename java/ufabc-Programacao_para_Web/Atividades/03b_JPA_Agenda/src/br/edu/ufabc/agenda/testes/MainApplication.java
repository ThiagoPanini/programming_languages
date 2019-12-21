package br.edu.ufabc.agenda.testes;

import java.util.List;
import java.util.Random;
import java.util.Scanner;

import javax.management.RuntimeErrorException;

import br.edu.ufabc.agenda.dao.CategoriaDAO;
import br.edu.ufabc.agenda.dao.ContatoDAO;
import br.edu.ufabc.agenda.modelo.Categoria;
import br.edu.ufabc.agenda.modelo.Contato;

public class MainApplication {

	public static void main(String[] args) {
		
		String nome, telefone, email;
		Categoria cat;
		Scanner sc = new Scanner(System.in);
		
		ContatoDAO dao = new ContatoDAO();
		CategoriaDAO ctgDao = new CategoriaDAO();
		
		/*Linhas abaixo serviram para inserção de Categorias no BD
		 * 
		 *
		 *String [] categorias = {"T.I", "Devs", "DBAs", "Backoffice", "Engenharia", "Data Science", "Dashboards",
				"Vendas", "Marketing", "Back-End", "Front-End", "Analytics"};
		ctgDao.insereTudo(categorias);
		
		List<Categoria> lista = ctgDao.getListaCategorias();
		for(Categoria item : lista) {
			item.statusCategoria();
		}*/
		
		//Menu para seleção do usuário
		System.out.println("\n- - - SISTEMA DE GERENCIAMENTO DE AGENDA - - -\n\n"
				+ "Escolha uma opção: \n"
				+ "[ 1 ] para inserir um Contato (Fornecer parâmetros)\n"
				+ "[ 2 ] para inserir um Contato (Parâmetros aleatórios)\n"
				+ "[ 3 ] para remover um Contato\n"
				+ "[ 4 ] para buscar um Contato pelo ID\n"
				+ "[ 5 ] para buscar Contato(s) pelo Nome\n"
				+ "[ 6 ] para visualizar a listagem de Contatos na Agenda");		
		Integer opt = Integer.parseInt(sc.nextLine());
		
		//Toma ação baseada no valor inserido pelo usuário
		switch(opt) {
		case 1:
			//Solicita dados para criação do Contato
			System.out.println("- - - CADASTRO DE CONTATO - - -");
			System.out.print("Nome: ");
			nome = sc.nextLine();
			System.out.print("Telefone: ");
			telefone = sc.nextLine();
			System.out.print("E-mail: ");
			email = sc.nextLine();
			System.out.println("Categoria: ");
			
			try {
				List<Categoria> cats = ctgDao.getListaCategorias();
				System.out.print("\n");
				for(Categoria categoria : cats) {
					categoria.statusCategoria();
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			
			System.out.println("\nSelecione o Id de uma das categorias abaixo ou [0] para inserir uma nova: ");
			Long optcat = Long.parseLong(sc.next());
			
			if (optcat == 0) {
				System.out.print("Insira uma nova Categoria: ");
				String newCat = sc.next();
				cat = new Categoria(newCat);
				ctgDao.insere(cat);
			} else {
				cat = ctgDao.buscaporId(optcat);
			}
			
			System.out.println("Categoria escolhida :" + cat.getId() + ", " + cat.getDescricao());
			Contato ct = new Contato(nome, telefone, email);
			ct.setCategoria(cat);
	
			try {
				dao.insere(ct);
				System.out.println("\nContato inserido na agenda"
						+ "\nNome: " + ct.getNome()
						+ "\nTelefone: " + ct.getTelefone()
						+ "\nE-mail: " + ct.getEmail()
						+ "\nCategoria: " + ct.getCategoria().getDescricao());	
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 2:
			//Cria dados aleatórios para inserção de Contato na Agenda
			Random rd = new Random();
			System.out.println("- - - GERANDO CONTATO ALEATÓRIO (TESTE) - - -");
			
			String [] nomes = {"Thiago Panini", "Axl Rose", "Slash", "David Gilmour", "Roger Waters",
					"Bill Gates", "Elon Musk", "Ronaldinho Gaúcho", "Ozielzinho", "Andy Timmons"};
			nome = nomes[rd.nextInt(nomes.length)];
			telefone = Integer.toString(rd.nextInt(99999999) + 1000000);
			email = "random@aleatorio.com.br";
			
			try {
				List<Categoria> cats = ctgDao.getListaCategorias();
				cat = cats.get(rd.nextInt(cats.size()));
				System.out.println("Categoria sorteada: ");
				
				cat.statusCategoria();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			
			Contato ct1 = new Contato(nome, telefone, email);
			ct1.setCategoria(cat);
			
			ct1.statusContato();
			
			try {
				dao.insere(ct1);
				System.out.println("\nContato inserido na agenda"
						+ "\nNome: " + ct1.getNome()
						+ "\nTelefone: " + ct1.getTelefone()
						+ "\nE-mail: " + ct1.getEmail()
						+ "\nCategoria: " + ct1.getCategoria().getDescricao());				
						
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 3:
			//Solicita inserção do id do Contato a ser removido
			System.out.println("- - - DELETANDO CONTATO - - -");
			System.out.println("Insira o id do contato que deseja deletar da Agenda: ");
			
			try {
				Contato contato = dao.buscaPorId(Long.parseLong(sc.nextLine()));
				dao.remove(contato);
				System.out.println("\nContato removido da agenda"
						+ "\nNome: " + contato.getNome()
						+ "\nTelefone: " + contato.getTelefone()
						+ "\nE-mail: " + contato.getEmail()
						+ "\nCategoria: " + contato.getCategoria().getDescricao());
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 4:
			//Solicita id do Contato a ser buscado
			System.out.println("- - - BUSCANDO UM CONTATO PELO ID - - -");
			System.out.println("Insira o Id do contato que deseja buscar na Agenda: ");
			
			try {
				Contato contato = dao.buscaPorId(Long.parseLong(sc.nextLine()));
				contato.statusContato();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 5:
			//Solicita nome do(s) Contato(s) a ser(em) buscado(s)
			System.out.println("- - - BUSCANDO CONTATO PELO NOME - - -");
			System.out.println("Insira o nome a ser buscado na Agenda: ");
			
			try {
				List<Contato> cts = dao.buscaPorNome(sc.nextLine());
				for (Contato contato : cts) {
					contato.statusContato();
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 6:
			//Imprime o stataus de todos os Contatos da Agenda
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
