package model;

import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class ConsumidorSMS extends Thread{
	
	//Atributos
	BlockingQueue<Mensagem> pilha;
	public static boolean unico = false;
	
	//M�todo Construtor
	public ConsumidorSMS(BlockingQueue<Mensagem> pilha){
		//- - - - PADR�O SINGLETON - - - -
		if (unico){
			throw new RuntimeException("Apenas uma instância permitida");
		}
		unico = true;
		this.pilha = pilha;
	}
	
	@Override
	public void run() {
		Random rd = new Random();
		for(int i=0; i<10; i++) {
			try {
				Mensagem m = pilha.take();
				System.out.println("Mensagem " + m.numero + " enviada do número " + m.celOrigem + " para o número " + m.celDestino);
				Thread.sleep((long) ((rd.nextDouble()*(0.8-0.2)+0.2)));
								
			} catch (Exception e){
				throw new RuntimeException(e);
			}
		}
	}

}
