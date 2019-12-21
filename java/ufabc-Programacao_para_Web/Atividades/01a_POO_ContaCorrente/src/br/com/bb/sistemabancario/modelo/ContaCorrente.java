package br.com.bb.sistemabancario.modelo;

import java.util.Random;
import java.util.Scanner;

public class ContaCorrente {
	
	//Atributos	
	private Integer id;
	private static Integer cont = 1;
	private String numero;
	private String agencia;
	private String descricao;
	private boolean ativa;
	private Integer variacao;
	
	//Método Construtor
	public ContaCorrente(String num, String ag, String desc, Integer var) {
		this.id = cont;
		cont++;
		this.numero = num;
		this.agencia = ag;
		this.descricao = desc;
		this.ativa = true;
		this.variacao = var;
	}
	
	//Construtor Vazio para Interação com Usuário
	public ContaCorrente() {
		this.id = cont;
		cont++;
	}
	
	//Getters e Setters	
	public Integer getId() {
		return id;
	}
	public void setId(Integer id) {
		this.id = id;
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
	public int getVariacao() {
		return variacao;
	}
	public void setVariacao(Integer variacao) {
		this.variacao = variacao;
	}	
	
	//Método para criar Conta Corrente
	public static ContaCorrente[] adicionaConta(Integer n) {
		
		Scanner sc = new Scanner(System.in);
		Random rd = new Random();
		ContaCorrente contas[] = new ContaCorrente[n];		
		for (int i=0; i<n; i++) {
			ContaCorrente cc = new ContaCorrente();
			System.out.println("\n- INSERINDO CONTA CORRENTE Nº " + (cont-1) + " -");
			System.out.print("Número da Conta: ");
			cc.setNumero(sc.nextLine());
			System.out.print("Agência Bancária: ");
			cc.setAgencia(sc.nextLine());
			System.out.print("Descrição da Conta: ");
			cc.setDescricao(sc.nextLine());
			cc.setAtiva(true);
			System.out.println("Conta ativada.");
			cc.setVariacao(rd.nextInt(5));	
			System.out.println("Variação: " + cc.getVariacao());
			
			contas[i] = cc;
		}
		return contas;		
	}

	
	//Método para mostrar Status da Conta Corrente
	public void statusConta() {
		System.out.println("\n- STATUS CONTA CORRENTE -");
		System.out.println("Id: " + this.getId());
		System.out.println("Número: " + this.getNumero());
		System.out.println("Agência: " + this.getAgencia());
		System.out.println("Descrição: " + this.getDescricao());
		System.out.println("Ativa: " + this.isAtiva());
		System.out.println("Variação: " + this.getVariacao());
	}

}