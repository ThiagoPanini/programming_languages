package com.projeto.projetoprogweb.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.projeto.projetoprogweb.model.dao.TipoNoticiaDao;
import com.projeto.projetoprogweb.model.entity.TipoNoticia;

@Service
public class TipoNoticiaService {

	//Implementando service REST
	@Autowired
	private TipoNoticiaDao tipoDao;
		
	//Procurar pelo Id
	public Optional<TipoNoticia> findById(Long id) {
		return tipoDao.findById(id);
	}
	//Retornar todos os tiponoticia
	public List<TipoNoticia> getAll(){
		return tipoDao.findAll();
	}
	//Criar tiponoticia
	public void create(TipoNoticia tipoNoticia) {
		tipoDao.save(tipoNoticia);
	}
	//Editar tiponoticia
	public void update(TipoNoticia tipoNoticia) {
		Optional<TipoNoticia> tipoNoticiaToUpdate = tipoDao.findById(tipoNoticia.getId_tipoNoticia());
		tipoNoticiaToUpdate.get().setNome(tipoNoticia.getNome());
		tipoDao.save(tipoNoticiaToUpdate.get());
	}
	//Deletar noticia
	public void delete(Long id) {
		tipoDao.deleteById(id);
	}
}
