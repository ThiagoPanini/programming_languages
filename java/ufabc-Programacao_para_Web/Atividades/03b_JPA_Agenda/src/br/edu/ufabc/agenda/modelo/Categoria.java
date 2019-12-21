package br.edu.ufabc.agenda.modelo;

import java.util.Set;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.OneToMany;
import javax.persistence.Table;

import org.hsqldb.lib.HashSet;

@Entity
@Table(name = "categoria")
public class Categoria {

	// Atributos
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long id_categoria;
	@Column(name = "descricao", nullable = false)
	private String descricao;
	@OneToMany(mappedBy = "categoria")
	private Set<Contato> contatos;
//	@OneToMany(targetEntity=Contato.class, mappedBy="categoria", cascade={CascadeType.ALL}, orphanRemoval=true)
//	private Set<Contato> contatos;

	// Método Construtor
	public Categoria(String desc) {
		this.descricao = desc;
	}

	// Método Construtor Vazio
	public Categoria() {
		// Vazio
	}

	// Getters e Setters
	public String getDescricao() {
		return descricao;
	}

	public void setDescricao(String descricao) {
		this.descricao = descricao;
	}

	public Set<Contato> getContato() {
		return contatos;
	}

	public void setContato(Set<Contato> contato) {
		this.contatos = contato;
	}

	public Long getId() {
		return id_categoria;
	}

	public void setId(Long id) {
		this.id_categoria = id;
	}

	// Método para apresentar Status das Categorias
	public void statusCategoria() {
		System.out.println("Id: " + this.getId() + " - " + this.getDescricao());
	}

}
