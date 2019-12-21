package br.edu.ufabc.agenda.dao;

import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

public class GeradorTabelas {

	public static void main(String[] args) {
		
		//Abrindo conexão com o BD e buscando infos do arquivo persistence.xml
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("agenda");
		System.out.println("Tabela gerada com sucesso!");
		factory.close();
	}

}
