package com.projeto.projetoprogweb.model.dao;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.projeto.projetoprogweb.model.entity.TipoNoticia;

@Repository
public interface TipoNoticiaDao extends JpaRepository<TipoNoticia, Long> {

}
