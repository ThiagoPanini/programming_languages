package br.edu.ufabc.agenda.dao;

import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

import br.edu.ufabc.agenda.modelo.Categoria;

public class CategoriaDAO {
	
	//--------------------------------- INSERE CATEGORIA NA AGENDA -----------------------------------------------
	public void insere(Categoria cat) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			manager.persist(cat);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
	}
	
		
	//--------------------------------- REMOVE CATEGORIA DA AGENDA -----------------------------------------------
	public void remove(Categoria cat) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		try {
			manager.getTransaction().begin();
			cat = manager.find(Categoria.class, cat.getId());
			manager.remove(cat);
			manager.getTransaction().commit();
		} finally {
			if (manager.getTransaction().isActive()) {
				manager.getTransaction().rollback();
			}
		}
	}
	
		
	//--------------------------------- BUSCA CATEGORIA POR ID --------------------------------------------
	public Categoria buscaporId(Long id) {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		Categoria cat = manager.find(Categoria.class, id);
		manager.close();
		return cat;
	}

		
	//--------------------------------- BUSCA CATEGORIA(S) POR NOME --------------------------------------------
	public List<Categoria> buscaPorNome(String nome){
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		
		@SuppressWarnings("unchecked")
		List<Categoria> cats = manager.createQuery("SELECT c from categoria c "
				+ "where descricao like '%"+nome+"%'").getResultList();
		
		manager.close();
		return cats;
	}
		
		
	//--------------------------------- LISTA TODAS AS CATEGORIAS -------------------------------------------------
	public List<Categoria> getListaCategorias() {
		
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		EntityManager manager = factory.createEntityManager();
		
		@SuppressWarnings("unchecked")
		List<Categoria> cats = manager.createQuery("select ct from Categoria ct").getResultList();
		
		manager.close();
		return cats;
	}
	
	//--------------------------------- INSERÇÃO INICIAL DE CATEGORIAS ----------------------------------------
		public void insereTudo(String [] cats) {
			
			EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
			EntityManager manager = factory.createEntityManager();
			try {
				for (int i=0; i<cats.length; i++) {
					Categoria categoria = new Categoria(cats[i]);
					manager.getTransaction().begin();
					manager.persist(categoria);
					manager.getTransaction().commit();
				}
			} finally {
				if (manager.getTransaction().isActive()) {
					manager.getTransaction().rollback();
				}
			}
		}

}
