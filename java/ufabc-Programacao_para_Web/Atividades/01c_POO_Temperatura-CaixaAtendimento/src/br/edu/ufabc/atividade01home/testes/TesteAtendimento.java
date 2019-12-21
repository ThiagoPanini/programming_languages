package br.edu.ufabc.atividade01home.testes;

import java.util.Random;

import br.edu.ufabc.atividade01home.modelo.CaixaAtendimento;

public class TesteAtendimento {

	public static void main(String[] args) {
		
		//Vetor para armazenar os 10 atendentes
		CaixaAtendimento[] caixas = new CaixaAtendimento[10];
		
		//Instanciando 10 caixas.
		for(int j=0; j<10; j++) {
			caixas[j] = new CaixaAtendimento();
		}
		
		Random rd = new Random();
		
		for (int i=0; i<300; i++){
			caixas[rd.nextInt(10)].chamaProximoFila();
		}
		
		System.out.println("- - - - - - - - - - - - - - - - - - -"
				+ " \nFIM DO EXPEDIENTE - VOLTE AMANHÃ!\n - - - - - - - - - - - - - - - - - - -");

	}

}
