package br.edu.ufabc.agenda.modelo;

import java.util.Random;

import org.hsqldb.persist.RandomAccessInterface;

public class Contato {
	
	//Atributos
	private Integer id;
	private String nome;
	private String email;
	private String endereço;
	
	//Método Construtor
	public Contato(String n, String e, String end) {
		super();
		this.nome = n;
		this.email = e;
		this.endereço = end;
	}

	//Getteres e Setters
	public Integer getId() {
		return id;
	}

	public void setId(Integer id) {
		this.id = id;
	}

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

	public String getEndereço() {
		return endereço;
	}

	public void setEndereço(String endereço) {
		this.endereço = endereço;
	}

	public void statusContato() {
		System.out.println("- STATUS DO CONTAO NA AGENDA -");
		System.out.println("Nome: " + this.getNome());
		System.out.println("E-mail: " + this.getEmail());
		System.out.println("Endereço: " + this.getEndereço());
		System.out.println("\n");
	}
	

}
