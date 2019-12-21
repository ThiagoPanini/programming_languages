package com.projeto.projetoprogweb.model.dao;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.projeto.projetoprogweb.model.entity.Comentario;

@Repository
public interface ComentarioDao extends JpaRepository<Comentario, Long> {

}
