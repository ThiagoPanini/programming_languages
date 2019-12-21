package br.edu.ufabc.statico.modelo;

public class CaixaAtendimento {
	
	//Atributos
	public static int cont = 0;
	public static int senha = 100;
	private int id;
	
		
	//Método Construtor
	public CaixaAtendimento() {
		cont++;
		this.id = cont;
	}
	
	public int getId(){
		return this.id;
	}
				
	//Métodos da Classe CaixaDeAtendimento	
	public void chamaProximoFila() {
		System.out.println("- - - - - - - - - -");
		System.out.println("Caixa: " + this.getId() + ": Próximo! \nSenha " + (senha) + "!");
		senha++;
	}
		
}
