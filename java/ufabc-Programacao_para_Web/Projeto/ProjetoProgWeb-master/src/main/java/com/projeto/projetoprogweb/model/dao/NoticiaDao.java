package com.projeto.projetoprogweb.model.dao;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.projeto.projetoprogweb.model.entity.Noticia;

@Repository
public interface NoticiaDao extends JpaRepository<Noticia, Long> {

}
