package com.projeto.projetoprogweb.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.projeto.projetoprogweb.model.dao.ComentarioDao;
import com.projeto.projetoprogweb.model.entity.Comentario;

@Service
public class ComentarioService {

	//Implementando service REST
	@Autowired
	private ComentarioDao comentDao;
	
	//Procurar pelo Id
	public Optional<Comentario> findById(Long id) {
		return comentDao.findById(id);
	}
	//Retornar todos os comentários
	public List<Comentario> getAll(){
		return comentDao.findAll();
	}
	//Criar comentário
	public void create(Comentario coment) {
		comentDao.save(coment);
	}
	//Editar comentário
	public void update(Comentario coment) {
		Optional<Comentario> comentToUpdate = comentDao.findById(coment.getId_comentario());
		comentToUpdate.get().setMensagem(coment.getMensagem());
		comentDao.save(comentToUpdate.get());
	}
	//Deletar comentário
	public void delete(Long id) {
		comentDao.deleteById(id);
	}
	
}
