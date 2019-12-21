package br.edu.ufabc.profissionais.testes;

import java.util.Random;

import br.edu.ufabc.profissionais.modelo.Empresario;
import br.edu.ufabc.profissionais.modelo.Politico;
import br.edu.ufabc.profissionais.modelo.Professor;
import br.edu.ufabc.profissionais.modelo.Profissional;

public class MainApplication {

	public static void main(String[] args) {
		
		Random rd = new Random();
		
		Profissional[] profissionais = new Profissional[9];
		
		//Criando objetos
		Politico toninho = new Politico("Toninho", "@lanchonete");
		Politico temey = new Politico ("Conde Temey", "@planalto");
		Politico trump = new Politico ("Pato Donaldo", "@foratrump");
		
		Professor fernando = new Professor("Fernando", "fernando@ufabc"); 
		Professor flavio = new Professor("Flávio", "flavio@horita");
		Professor vera = new Professor("Vera", "vera@ufabc");
		
		Empresario batista = new Empresario("Eike", "eike@batista");
		Empresario bob = new Empresario("Bob Fischer", "fisher@xess");
		Empresario magnus = new Empresario("Magnus", "@magnus@xess");
		
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
		
		for (int i=0; i<profissionais.length; i++){
			int aleatorio = rd.nextInt(4);
			for (int j=0; j<=aleatorio; j++) {
				profissionais[i].adicionaContatos(profissionais[rd.nextInt(9)]);
			}
		}
		
		for (int i=0; i<profissionais.length; i++) {
			profissionais[i].mostraContatos();
		}
	}

}

