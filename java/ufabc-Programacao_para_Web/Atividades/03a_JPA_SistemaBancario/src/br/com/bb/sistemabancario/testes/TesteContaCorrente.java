package br.com.bb.sistemabancario.testes;

import java.util.List;
import java.util.Random;
import java.util.Scanner;

import br.com.bb.sistemabancario.dao.ContaCorrenteDAO;
import br.com.bb.sistemabancario.modelo.ContaCorrente;

public class TesteContaCorrente {

	public static void main(String[] args) {
		
		String num, ag, desc;
		Integer var;
		Scanner sc = new Scanner(System.in);
		
		ContaCorrenteDAO dao = new ContaCorrenteDAO();
		
		//Menu para seleção do usuário
		System.out.println("- - - SISTEMA DE GERENCIAMENTO BANCÁRIO - - -\n\n"
				+ "Escolha uma opção: \n"
				+ "[ 1 ] para inserir uma Conta (Fornecer parâmetros)\n"
				+ "[ 2 ] para inserir uma Conta (Parâmetros aleatórios)\n"
				+ "[ 3 ] para remover uma Conta\n"
				+ "[ 4 ] para buscar uma Conta pelo Id\n"
				+ "[ 5 ] para visualizar a listagem de Contas no Banco de Dados");		
		Integer opt = Integer.parseInt(sc.nextLine());
		
		//Toma ação baseada no valor inserido pelo usuário
		switch(opt) {
		case 1:
			//Solicita dados para criação da CC
			System.out.println("- - - CADASTRO DE CONTA CORRENTE - - -");
			System.out.print("Número: ");
			num = sc.nextLine();
			System.out.print("Agência: ");
			ag = sc.nextLine();
			System.out.print("Descrição: ");
			desc = sc.nextLine();
			System.out.print("Variação: ");
			var = Integer.parseInt(sc.nextLine());
			
			ContaCorrente cc = new ContaCorrente(num, ag, desc, var);
	
			try {
				dao.insere(cc);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 2:
			//Cria dados aleatórios para inserção de CC no BD
			Random rd = new Random();
			System.out.println("- - - GERANDO CONTA CORRENTE ALEATÓRIA (TESTE) - - -");
			num = Integer.toString(rd.nextInt(999999));
			ag = Integer.toString(rd.nextInt(9999));
			String [] descricao = {"Banco do Brasil", "Bradesco", "Itaú", "Nubank", "Banco Inter", "Santander"};
			desc = "Conta do " + descricao[rd.nextInt(5)];
			var = rd.nextInt(6);
			
			ContaCorrente cc1 = new ContaCorrente(num, ag, desc, var);

			try {
				dao.insere(cc1);
				System.out.println("Conta Corrente inserida\n"
						+ "Número: " + cc1.getNumero()
						+ "\nAgência: " + cc1.getAgencia()
						+ "\nDescrição: " + cc1.getDescricao()
						+ "\nVariação: " + cc1.getVariação());
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 3:
			//Solicita inserção do id da CC a ser removida
			System.out.println("- - - DELETANDO CONTA CORRENTE - - -");
			System.out.println("Insira o id da conta que deseja deletar do Banco: ");
			
			try {
				ContaCorrente conta = dao.buscaPorId(Long.parseLong(sc.nextLine()));
				dao.remove(conta);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 4:
			//Solicita id da CC a ser buscada
			System.out.println("- - - BUSCANDO UMA CONTA CORRENTE PELO ID - - -");
			System.out.println("Insira o id da conta que deseja buscar no Banco: ");
			
			try {
				ContaCorrente conta = dao.buscaPorId(Long.parseLong(sc.nextLine()));
				conta.statusContaCorrente();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
			
		case 5:
			//Imprime o stataus de todas Contas presentes no BD
			System.out.println("- - - MOSTRANDO TODAS AS CONTAS DO BANCO - - -");
			try {
				List<ContaCorrente> ccs = dao.getListaContas();
				for (ContaCorrente conta : ccs) {
					conta.statusContaCorrente();
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			break;
		default: System.out.println("Entrada inválida.");
			
		}
		
	}

}
