package br.edu.ufabc.agenda.dao;

import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

import br.edu.ufabc.agenda.modelo.Contato;

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


public class ContatoDAO {
	
	//--------------------------------- INSERE CONTATO NA AGENDA -----------------------------------------------
	public void insere(Contato ct) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda"); //Referência persistence unit
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			manager.persist(ct);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
		manager.close();
	}
	
	//--------------------------------- REMOVE CONTATO DA AGENDA -----------------------------------------------
	public void remove (Contato ct) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			ct = manager.find(Contato.class, ct.getId()); //Encontra o valor do Id
			manager.remove(ct);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
		manager.close();
	}
	
	//--------------------------------- BUSCA CONTATO POR ID --------------------------------------------
	public Contato buscaPorId (Long id) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		Contato ct = manager.find(Contato.class, id); //Encontra o valor do Id
		manager.close();
		return ct;
	}
	
	//--------------------------------- BUSCA CONTATO(S) POR NOME --------------------------------------------
	public List<Contato> buscaPorNome (String nome) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		
		@SuppressWarnings("unchecked")
		List<Contato> cts = manager.createQuery("SELECT ct from Contato ct where nome like '%"+nome+"%'").getResultList();
		
		manager.close();
		return cts;
	}
	
	
	//--------------------------------- LISTA TODOS OS CONTATOS -------------------------------------------------
	public List<Contato> getListaContatos() {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		
		@SuppressWarnings("unchecked")
		List<Contato> cts = manager.createQuery("SELECT ct from Contato ct").getResultList();
		
		manager.close();
		return cts;
		
	}
	
	
}