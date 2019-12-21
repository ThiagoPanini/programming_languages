package br.edu.ufabc.mensagem.modelo;

public class Mensagem {
	
	//Atributos	
	String texto, celOrigem, celDestino;
	int numero;
	static int cont = 1;
	
	//Método Construtor
	public Mensagem(String t, String origem, String destino) {
		
		this.texto = t;
		this.celOrigem = origem;
		this.celDestino = destino;
		synchronized(this){
			numero = cont++;
		}
	}
	
	//Getters e Setters
	public String getTexto() {
		return texto;
	}

	public void setTexto(String texto) {
		this.texto = texto;
	}

	public String getCelOrigem() {
		return celOrigem;
	}

	public void setCelOrigem(String celOrigem) {
		this.celOrigem = celOrigem;
	}

	public String getCelDestino() {
		return celDestino;
	}

	public void setCelDestino(String celDestino) {
		this.celDestino = celDestino;
	}
	
	//
	
	public void statusMensagem() {
		System.out.println("\n- - - Status Mensagem - - -");
		System.out.println("Id: " + numero);
		System.out.println("Celular de Origem: " + this.getCelOrigem());
		System.out.println("Celular de Destino: " + this.getCelDestino());
		System.out.println("Texto: " + this.getTexto());
	}

}