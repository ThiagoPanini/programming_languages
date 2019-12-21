package br.com.bb.sistemabancario.dao;

import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

import br.com.bb.sistemabancario.modelo.ContaCorrente;

/*
 * - - - - - - - - - - - - - - - - - - - - - - CLASSES JPA - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * 
 * EntityManagerFactory: faz a conexão com o BD com as propriedades definidas no arquivo persistence.xml e
 * também referencia os objetos do tipo EntityManager.
 * 
 * EntityManager: oferece métodos para executar as operações (CRUD) no BD. Uma instância é obtida através
 * do objeto EntityManagerFactory.
 * 
 * EntityTransaction: gerencia as transações (toda operação que modifica o BD (CRUD) devem ser executadas
 * dentro de uma transação.
 * 
 * - - - - - - - - - - - - - - - - - - - - - - - OPERAÇÕES - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * 
 * persist(objeto): salva um objeto no BD.
 * remove(objeto): remove o objeto (registro) do BD
 * merge(objeto): altera um registro no BD
 * find(class, id): busca um objeto no BD de acordo com o Id e classe do objeto 			- 	
 */


public class ContaCorrenteDAO {
	
	//--------------------------------- INSERE CONTA CORRENTE -----------------------------------------------
	public void insere(ContaCorrente cc) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("sistemabancario"); //Referência persistence unit
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			manager.persist(cc);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
		manager.close();
	}
	
	//--------------------------------- REMOVE CONTA CORRENTE -----------------------------------------------
	public void remove (ContaCorrente cc) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("sistemabancario");
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			cc = manager.find(ContaCorrente.class, cc.getId()); //Encontra o valor do Id
			manager.remove(cc);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
		manager.close();
	}
	
	//--------------------------------- BUSCA CONTA CORRENTE POR ID --------------------------------------------
	public ContaCorrente buscaPorId (Long id) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("sistemabancario");
		EntityManager manager = factory.createEntityManager();
		ContaCorrente cc = manager.find(ContaCorrente.class, id); //Encontra o valor do Id
		manager.close();
		return cc;
	}
	
	//--------------------------------- LISTA TODAS AS CONTAS -------------------------------------------------
	public List<ContaCorrente> getListaContas() {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("sistemabancario");
		EntityManager manager = factory.createEntityManager();
		
		@SuppressWarnings("unchecked")
		List<ContaCorrente> ccs = manager.createQuery("SELECT cc from ContaCorrente cc").getResultList();
		
		manager.close();
		return ccs;
		
	}
	
	
}