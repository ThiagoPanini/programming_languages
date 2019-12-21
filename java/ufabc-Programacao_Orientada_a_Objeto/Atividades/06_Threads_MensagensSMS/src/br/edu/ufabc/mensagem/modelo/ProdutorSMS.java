package br.edu.ufabc.mensagem.modelo;

import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class ProdutorSMS extends Thread {
	
	//Atributos
	BlockingQueue<Mensagem> pilha;
	int num;
	
	//Método Construtor
	public ProdutorSMS(int n, BlockingQueue<Mensagem> pilha) {
		this.num = n;
		this.pilha = pilha;
	}
	
	@Override
	public void run(){
		Random rd = new Random();
		for(int i=0; i<10; i++){
			Mensagem m = new Mensagem("Teste", Integer.toString(rd.nextInt(99999999)+10000000), Integer.toString(rd.nextInt(99999999)+10000000));
			try {
				System.out.println("Mensagem " + m.numero + " criada");
				pilha.put(m);
				Thread.sleep((long) ((rd.nextDouble()*(1.4-0.6) + 0.6)));
				
			} catch (Exception e){
				throw new RuntimeException(e);
			}
		}
	}	

}
