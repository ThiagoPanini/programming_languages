package br.edu.ufabc.mensagem.testes;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import br.edu.ufabc.mensagem.modelo.ConsumidorSMS;
import br.edu.ufabc.mensagem.modelo.Mensagem;
import br.edu.ufabc.mensagem.modelo.ProdutorSMS;

public class MainApplication {

	public static void main(String[] args) {
		
		BlockingQueue<Mensagem> filaMensagens = new ArrayBlockingQueue<Mensagem>(100);
		
		ProdutorSMS p1 = new ProdutorSMS(1, filaMensagens);
		ProdutorSMS p2 = new ProdutorSMS(2, filaMensagens);
		ProdutorSMS p3 = new ProdutorSMS(3, filaMensagens);
		
		ConsumidorSMS c1 = new ConsumidorSMS(filaMensagens);
		
		p1.start();
		p2.start();
		p3.start();
		
		c1.start();
		
		

	}

}
