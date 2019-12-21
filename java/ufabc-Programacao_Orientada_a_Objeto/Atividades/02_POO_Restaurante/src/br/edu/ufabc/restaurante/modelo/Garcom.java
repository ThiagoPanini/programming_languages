package br.edu.ufabc.restaurante.modelo;

public class Garcom {
	
	//Atributos
	private Integer cod_garcom;
	private String nome;
	private Boolean disponivel;
	
	//Método Construtor
	public Garcom(Integer c, String n){
		this.cod_garcom = c;
		this.nome = n;
		this.disponivel = true;
	}
	
	//Métodos Especiais (Getters e Setters)
	public Integer getCod_garcom() {
		return cod_garcom;
	}

	public void setCod_garcom(Integer cod_garcom) {
		this.cod_garcom = cod_garcom;
	}

	public String getNome() {
		return nome;
	}

	public void setNome(String nome) {
		this.nome = nome;
	}

	public Boolean getDisponivel() {
		return disponivel;
	}

	public void setDisponivel(Boolean disponivel) {
		this.disponivel = disponivel;
	}
	
	//Métodos da Classe Garcom
	
	public void statusGarcom(){
		System.out.println("\n----- Status do Garçom " + this.getCod_garcom() + " -----");
		System.out.println("Nome: " + this.getNome());
		System.out.println("Disponível? " + this.getDisponivel());		
	}

	public void atenderMesa(Mesa mesa){
		this.setDisponivel(false);
	}
	
	
}
