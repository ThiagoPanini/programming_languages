package br.edu.ufabc.profissionais.testes;

import java.util.Random;
import java.util.Scanner;

import br.edu.ufabc.profissionais.modelo.Empresario;
import br.edu.ufabc.profissionais.modelo.Politico;
import br.edu.ufabc.profissionais.modelo.Professor;
import br.edu.ufabc.profissionais.modelo.Profissional;

public class MainApplication {

	public static void main(String[] args) {
		
		Random rd = new Random();
		Scanner sc = new Scanner(System.in);
		
		Profissional[] profissionais = new Profissional[9];
		
		//Criando objetos
		Politico toninho = new Politico("Vereador", "vereador@gov.com.br");
		Politico temey = new Politico ("Prefeito", "prefeito@gov.com.br");
		Politico trump = new Politico ("Deputado", "deputado@gov.com.br");
		
		Professor fernando = new Professor("Fernando", "fernando@ufabc.edu.br"); 
		Professor flavio = new Professor("Flávio", "flavio@ufabc.edu.br");
		Professor vera = new Professor("Vera", "vera@ufabc.edu.br");
		
		Empresario batista = new Empresario("Thiago", "panini@panini.com.br");
		Empresario bob = new Empresario("Bob Fischer", "fisher@xess.com");
		Empresario magnus = new Empresario("Magnus", "@magnus@xess.com");
		
		//Adicionando objetos ao vetor
		profissionais[0] = toninho;
		profissionais[1] = temey;
		profissionais[2] = trump;
		profissionais[3] = fernando;
		profissionais[4] = flavio;
		profissionais[5] = vera;
		profissionais[6] = batista;
		profissionais[7] = bob;
		profissionais[8] = magnus;
		
		System.out.println("- - - SISTEMA DE AGENDA PROFISSIONAL - - -\n");
		System.out.println("Escolha uma opção: \n"
				+ "[ 1 ] para adicionar um contato\n"
				+ "[ 2 ] para mostrar contatos");
		Integer opt1 = Integer.parseInt(sc.nextLine());
		
		switch(opt1) {		
		case 1:
			System.out.println("\n- - - ADICIONANDO CONTATOS - - -\n");
			System.out.println("Selecione uma opçao: \n"
					+ "[ 1 ] adicionar contatos aleatoriamente em todas as agendas\n"
					+ "[ 2 ] adicionar contatos específicos em agendas específicas");
			Integer opt2 = Integer.parseInt(sc.nextLine());
			
			switch(opt2) {
			case 1:
				for (int i=0; i<profissionais.length; i++){
					int aleatorio = rd.nextInt(4);
					for (int j=0; j<=aleatorio; j++) {
						profissionais[i].adicionaContatos(profissionais[rd.nextInt(9)]);
					}
				}
				
			case 2:
				System.out.println("\n- - - AGENDAS DISPONÍVEIS - - -");
				for (int i=1; i<profissionais.length+1; i++) {
					System.out.println("- - - - - - - - -");
					System.out.print("Agenda " + i);
					profissionais[i-1].mostraNomes();
				}
				System.out.println("\nSelecione uma agenda: ");
				Integer opt3 = Integer.parseInt(sc.nextLine());
				
				System.out.println("\n- - - AGENDA DE " + profissionais[opt3-1].getNome() + " SELECIONADA - - -");
				System.out.println("Selecione, através dos números mostrados acima,"
						+ " o contato que deseja adicionar a agenda do(a) "
						+ profissionais[opt3-1].getClass().getSimpleName() + "(a) " 
						+ profissionais[opt3-1].getNome());
				while (true) {
					Integer id = Integer.parseInt(sc.nextLine());
					if (id == 0) {
						break;
					}
					try {
						profissionais[opt3-1].adicionaContatos(profissionais[id-1]);
						System.out.println("Contato adicionado: " + profissionais[id-1].getClass().getSimpleName() + 
								" " + profissionais[id-1].getNome());
						System.out.println("\nSelecione outro contato ou 0 para sair: ");
					} catch (ArrayIndexOutOfBoundsException index) {
							System.out.println("Contato não existente! Insira um item válido! ");
						}
					}					
				default: System.out.println("Valor inválido.");	
				}
			}
				
		for (int i=0; i<profissionais.length; i++) {
			profissionais[i].mostraContatos();
		}
	}

}

