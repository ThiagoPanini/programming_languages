package com.projeto.projetoprogweb.model.entity;

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

@Entity
@Table(name="comentario")
public class Comentario {
	
	//Atributos da tabela comentario
	@Id
	@GeneratedValue(strategy=GenerationType.IDENTITY)
	private Long id_comentario;
	@Column(name="nome", nullable=false)
	private String nome;
	@Column(name="mensagem", nullable=false)
	private String mensagem;
	@Column(name="data")
	private String data;
	@ManyToOne(fetch = FetchType.LAZY, cascade=CascadeType.ALL) 
	@JoinColumn(name = "id_noticia") //comentario muitos pra um noticia
	private Noticia noticia;
	
	public Comentario() {
		
	}
	public Comentario(String dataAtual) {
		this.data = dataAtual;
	}
	
	//Getters e Setters
	public String getNome() {
		return nome;
	}
	public void setNome(String nome) {
		this.nome = nome;
	}
	public String getMensagem() {
		return mensagem;
	}
	public void setMensagem(String mensagem) {
		this.mensagem = mensagem;
	}
	public Noticia getNoticia() {
		return noticia;
	}
	public void setNoticia(Noticia noticia) {
		this.noticia = noticia;
	}
	public Long getId_comentario() {
		return id_comentario;
	}
	public String getData() {
		return data;
	}
	public void setData(String data) {
		this.data = data;
	}	
	
	
}
