package com.projeto.projetoprogweb.model.entity;

import java.util.Set;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Table;

@Entity
@Table(name="noticia")
public class Noticia {

	//Atributos da tabela noticia
	@Id
	@GeneratedValue(strategy=GenerationType.IDENTITY)
	private Long id_noticia;	
	@Column(name="titulo", nullable=false)
	private String titulo;
	@Column(name="subtitulo", nullable=false)
	private String subtitulo;
	@Column(name="url", nullable=false)
	private String url;
	@ManyToOne(fetch = FetchType.LAZY, cascade=CascadeType.ALL) 
	@JoinColumn(name = "id_tipoNoticia") //noticia muitos pra um tipo
	private TipoNoticia tipoNoticia;
	@OneToMany(mappedBy="noticia")//noticia um pra muitos comentario
	private Set<Comentario> comentario;
	
	
	//Getters e Setters
	public String getTitulo() {
		return titulo;
	}
	public void setTitulo(String titulo) {
		this.titulo = titulo;
	}
	public String getSubtitulo() {
		return subtitulo;
	}
	public void setSubtitulo(String subtitulo) {
		this.subtitulo = subtitulo;
	}
	public String getUrl() {
		return url;
	}
	public void setUrl(String url) {
		this.url = url;
	}
	public TipoNoticia getTipoNoticia() {
		return tipoNoticia;
	}
	public void setTipoNoticia(TipoNoticia tipoNoticia) {
		this.tipoNoticia = tipoNoticia;
	}
	public Long getId_noticia() {
		return id_noticia;
	}
	public Set<Comentario> getComentario() {
		return comentario;
	}
	public void setComentario(Set<Comentario> comentario) {
		this.comentario = comentario;
	}
	
	
}
