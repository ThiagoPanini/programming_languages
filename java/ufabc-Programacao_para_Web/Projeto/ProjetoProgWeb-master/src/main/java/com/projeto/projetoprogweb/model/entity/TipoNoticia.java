package com.projeto.projetoprogweb.model.entity;

import java.util.Set;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.OneToMany;
import javax.persistence.Table;

@Entity
@Table(name="tiponoticia")
public class TipoNoticia {

	//Atributos da tabela tiponoticia
	@Id
	@GeneratedValue(strategy=GenerationType.IDENTITY)
	private Long id_tipoNoticia;
	@Column(name="nome", nullable=false)
	private String nome;
	@OneToMany(mappedBy="tipoNoticia")//tiponoticia um pra muitos noticia
	private Set<Noticia> noticia;
	
	//Getters e Setters
	public Set<Noticia> getNoticia() {
		return noticia;
	}
	public void setNoticia(Set<Noticia> noticia) {
		this.noticia = noticia;
	}
	public Long getId_tipoNoticia() {
		return id_tipoNoticia;
	}
	public String getNome() {
		return nome;
	}
	public void setNome(String nome) {
		this.nome = nome;
	}

}
