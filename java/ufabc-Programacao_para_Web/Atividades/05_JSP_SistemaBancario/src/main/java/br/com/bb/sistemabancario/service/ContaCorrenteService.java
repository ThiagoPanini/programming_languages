package br.com.bb.sistemabancario.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import br.com.bb.sistemabancario.model.dao.ContaCorrenteDAO;
import br.com.bb.sistemabancario.model.entity.ContaCorrente;

/*@Service = Regras de negócio
 * 
 * Camada responsável por implementar as tomadas de decisão, determinando o que será gravado e exibido.
 * - Valicações
 * - Condições
 * - Formatações
 * - etc...
 * 
 * Trabalha em conjunto com a DAO e encapsula as operações básicas CRUD (Create, Read, Update e Delete).
 */

@Service 
public class ContaCorrenteService {
	
	//- - - - IMPLEMENTANDO SERVICE REST - - - -
	@Autowired
	private ContaCorrenteDAO contacorrenteDao;

	/* MÉTODOS RESPONSÁVEIS POR GERENCIAR O ACESSO AO BD */
	
	//- - - - CRIA CONTA CORRENTE - - - -
	public void create(ContaCorrente cc) {
		contacorrenteDao.save(cc);
	}
	
	//- - - - PROCURA CONTA CORRENTE PELO ID - - - -
	public Optional<ContaCorrente> findById(Long id) {
		return contacorrenteDao.findById(id);
	}
	
	//- - - - RETORNA TODAS AS CONTAS - - - -
	public List<ContaCorrente> getAll() {
		return contacorrenteDao.findAll();
	}
	
	//- - - - EDITAR CONTA CORRENTE - - - -
	public void update(ContaCorrente cc) {
		Optional<ContaCorrente> ccToUpdate = contacorrenteDao.findById(cc.getId());
		ccToUpdate.get().setAgencia(cc.getAgencia());
		ccToUpdate.get().setNumero(cc.getNumero());
		ccToUpdate.get().setDescricao(cc.getDescricao());
		ccToUpdate.get().setAtiva(cc.isAtiva());
		ccToUpdate.get().setVariacao(cc.getVariacao());
	}
	
	//- - - - DELETAR UMA CONTA CORRENTE - - - -	
	public void delete(Long id) {
		contacorrenteDao.deleteById(id);
	}
	

}
