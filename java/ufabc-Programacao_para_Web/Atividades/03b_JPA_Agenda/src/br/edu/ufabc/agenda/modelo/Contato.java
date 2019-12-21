																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		package br.edu.ufabc.agenda.modelo;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;

@Entity //Indica que a classe Contato é uma entidade
@Table(name="contato") //Define o nome da tabela [opcional]
public class Contato {
	
	//Atributos
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY) //serial ou auto_increment
	private Long id;
	@Column(name="nome", nullable=false)
	private String nome;
	@Column(name="telefone", nullable=false)
	private String telefone;
	@Column(name="email", nullable=false)
	private String email;
	@ManyToOne(fetch = FetchType.LAZY) //cascade=CascadeType.ALL
	@JoinColumn(name = "id_categoria")
	private Categoria categoria;
//	@ManyToOne(targetEntity=Categoria.class)
//	@JoinColumn(name="id")
//	private Categoria categoria;
	
	//Método Construtor	
	public Contato(String nome, String telefone, String email) {
		super();
		this.nome = nome;
		this.telefone = telefone;
		this.email = email;
		//this.categoria = categoria;
	}

	//Método Construtor vazio para auxiliar em alguns métodos
	public Contato() {
		//Empty
	}
	
	//Getters e Setters
	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public String getNome() {
		return nome;
	}

	public void setNome(String nome) {
		this.nome = nome;
	}

	public String getTelefone() {
		return telefone;
	}

	public void setTelefone(String telefone) {
		this.telefone = telefone;
	}

	public String getEmail() {
		return email;
	}

	public void setEmail(String email) {
		this.email = email;
	}
	
	public Categoria getCategoria() {
		return categoria;
	}

	public void setCategoria(Categoria categoria) {
		this.categoria = categoria;
	}

	
	//Método para apresentar status do Contato
	public void statusContato() {
		System.out.println("\n- STATUS CONTATO -");
		System.out.println("Id: " + this.getId());
		System.out.println("Nome: " + this.getNome());
		System.out.println("Telefone: " + this.getTelefone());
		System.out.println("E-mail: " + this.getEmail());
		System.out.println("Categoria: " + this.getCategoria().getDescricao());
	}
	
	
}