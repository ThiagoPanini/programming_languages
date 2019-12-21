package com.projeto.projetoprogweb.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.projeto.projetoprogweb.model.dao.NoticiaDao;
import com.projeto.projetoprogweb.model.entity.Noticia;

@Service
public class NoticiaService {

	//Implementando service REST
	@Autowired
	private NoticiaDao noticiaDao;
	
	//Procurar pelo Id
	public Optional<Noticia> findById(Long id) {
		return noticiaDao.findById(id);
	}
	//Retornar todos os noticia
	public List<Noticia> getAll(){
		return noticiaDao.findAll();
	}
	//Criar noticia
	public void create(Noticia noticia) {
		noticiaDao.save(noticia);
	}
	//Editar noticia
	public void update(Noticia noticia) {
		Optional<Noticia> noticiaToUpdate = noticiaDao.findById(noticia.getId_noticia());
		noticiaToUpdate.get().setTitulo(noticia.getTitulo());
		noticiaToUpdate.get().setSubtitulo(noticia.getSubtitulo());
		noticiaToUpdate.get().setUrl(noticia.getUrl());
		noticiaDao.save(noticiaToUpdate.get());
	}
	//Deletar noticia
	public void delete(Long id) {
		noticiaDao.deleteById(id);
	}
}
