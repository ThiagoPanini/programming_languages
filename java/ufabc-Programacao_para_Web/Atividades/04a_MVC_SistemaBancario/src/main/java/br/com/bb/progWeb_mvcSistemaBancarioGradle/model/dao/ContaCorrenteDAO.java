package br.com.bb.progWeb_mvcSistemaBancarioGradle.model.dao;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import br.com.bb.progWeb_mvcSistemaBancarioGradle.model.entity.ContaCorrente;

@Repository
public interface ContaCorrenteDAO extends JpaRepository<ContaCorrente, Long> {

}
