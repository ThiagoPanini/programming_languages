package br.com.bb.progWeb_mvcSistemaBancarioGradle.model.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity //Indica que a classe Conta Corrente é uma entidade
@Table(name="contacorrente") //Define o nome da tabela [opcional]
public class ContaCorrente {
	
	//Atributos
	@Id //Indica que o atributo é chave primária da tabela
	@GeneratedValue(strategy = GenerationType.IDENTITY) //Define que o atributo será serial (auto_increment)
	private Long id;
	@Column(name="numero", nullable=false) //Define o nome da coluna [opcional]
	private String numero;
	@Column(name="agencia", nullable=false)
	private String agencia;
	@Column(name="descricao", nullable=false)
	private String descricao;
	@Column(name="ativa")
	private boolean ativa;
	@Column(name="variacao")
	private Integer variação;
	
//	//Método Construtor
//	public ContaCorrente(String numero, String agencia, String descricao, Integer variação) {
//		super();
//		this.numero = numero;
//		this.agencia = agencia;
//		this.descricao = descricao;
//		this.ativa = true;
//		this.variação = variação;
//	}
//	
//	//Método Construtor alternativo e vazio para funcionar alguns métodos que referenciam ContaCorrente
//	public ContaCorrente() {
//		
//	}
	
	//Getters e Setters
	public Long getId() {
		return id;
	}

	public void setId(Long id) {
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

	public Integer getVariação() {
		return variação;
	}

	public void setVariação(Integer variação) {
		this.variação = variação;
	}
	
//	//Método para apresentar a Conta Corrente
//	public void statusContaCorrente() {
//		System.out.println("- STATUS CONTA CORRENTE -");
//		System.out.println("Id: " + this.getId());
//		System.out.println("Número: " + this.getNumero());
//		System.out.println("Agência: " + this.getAgencia());
//		System.out.println("Descrição: " + this.getDescricao());
//		System.out.println("Ativa: " + this.isAtiva());
//		System.out.println("Variação: " + this.getVariação());
//		System.out.println("\n");
//	}
	
	
	
	
}