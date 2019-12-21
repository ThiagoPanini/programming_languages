package br.com.bb.sistemabancario.modelo;

import java.util.Random;

import org.hsqldb.persist.RandomAccessInterface;

public class ContaCorrente {
	
	//Atributos
	private Integer id;
	private static Integer cont=2;
	private String numero;
	private String agencia;
	private String descricao;
	private boolean ativa;
	private Integer variação;
	
	//Método Construtor
	public ContaCorrente(String num, String ag, String desc, Integer var) {
		super();
		this.id = cont;
		cont++;
		this.numero = num;
		this.agencia = ag;
		this.descricao = desc;
		this.ativa = true;
		this.variação = var;
	}
	
	//Getters e Setters
	public Integer getId() {
		return id;
	}

	public void setId(Integer id) {
		this.id = id;
	}

	public static Integer getCont() {
		return cont;
	}

	public static void setCont(Integer cont) {
		ContaCorrente.cont = cont;
	}

	public String getNumero() {
		return numero;
	}

	public void setNumero(String numero) {
		this.numero = numero;
	}

	public String getAgencia() {
		return agencia;
	}

	public void setAgencia(String agencia) {
		this.agencia = agencia;
	}

	public String getDescricao() {
		return descricao;
	}

	public void setDescricao(String descricao) {
		this.descricao = descricao;
	}

	public boolean isAtiva() {
		return ativa;
	}

	public void setAtiva(boolean ativa) {
		this.ativa = ativa;
	}

	public Integer getVariação() {
		return variação;
	}

	public void setVariação(Integer variação) {
		this.variação = variação;
	}
	
	//Método para apresentar a Conta Corrente
	public void statusContaCorrente() {
		System.out.println("- STATUS CONTA CORRENTE -");
		System.out.println("Id: " + this.getId());
		System.out.println("Número: " + this.getNumero());
		System.out.println("Agência: " + this.getAgencia());
		System.out.println("Descrição: " + this.getDescricao());
		System.out.println("Ativa: " + this.isAtiva());
		System.out.println("Variação: " + this.getVariação());
		System.out.println("\n");
	}
	

}
