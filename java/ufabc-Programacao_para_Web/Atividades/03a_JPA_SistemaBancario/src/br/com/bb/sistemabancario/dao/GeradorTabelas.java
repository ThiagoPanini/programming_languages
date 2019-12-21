package br.com.bb.sistemabancario.dao;

import javax.persistence.EntityManagerFactory;
import javax.persistence.Persistence;

public class GeradorTabelas {

	public static void main(String[] args) {
		
		//Abrindo conexão com o BD e buscando infos do arquivo persistence.xml
		EntityManagerFactory factory = Persistence.createEntityManagerFactory("sistemabancario");
		System.out.println("Tabela gerada com sucesso!");
		factory.close();
	}

}
