package br.edu.ufabc.profissionais.modelo;

import java.util.ArrayList;

public abstract class Profissional {
	
	protected ArrayList<Profissional> contatos = new ArrayList<Profissional>();
	
	//Atributos
	private String nome;
	private String email;
	
	//Getters e Setters
	public String getNome() {
		return nome;
	}
	public void setNome(String nome) {
		this.nome = nome;
	}
	public String getEmail() {
		return email;
	}
	public void setEmail(String email) {
		this.email = email;
	}
	
	//Métodos
	public void adicionaContatos(Profissional p){
		
		try {
			contatos.add(p);
		} 
		
		catch (NullPointerException n) {
			System.out.println("Valor nulo é inválido. Não se trata de um Profissional!");
			}
		
		if (p.equals(this)) {
			throw new RuntimeException("Não é possível adicionar a si mesmo.");
		}
		
		p.contatos.add(this);
	}
	
	public void mostraContatos() {
		int i = 1;
		
		System.out.println("- - - - - - - - - - - - - - - - - - - - - - - -");
		if ("Politico".equals(this.getClass().getSimpleName())){
			System.out.println("CONTATOS do(a) Político(a) " + this.getNome());
		}
		
		if ("Professor".equals(this.getClass().getSimpleName())){
			System.out.println("CONTATOS do(a) Professor(a) " + this.getNome());
		}
		
		if ("Empresario".equals(this.getClass().getSimpleName())){
			System.out.println("CONTATOS do(a) Empresário(a) " + this.getNome());
		}	
			
		for (Profissional contato : contatos){
			System.out.println("- - - - - - - - - - - - - - - - - - - - - - - -");
			System.out.println("Contato " + i);
			System.out.println(contato.getClass().getSimpleName());
			System.out.println("Nome: " + contato.getNome());
			System.out.println("E-mail: " + contato.getEmail());
			i++;
		}
	}
	
	public void mostraNomes() {		
		System.out.println("\nNome: " + this.getNome());
		System.out.println("Cargo: " + this.getClass().getSimpleName());
	}
	
}
