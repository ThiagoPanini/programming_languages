package br.edu.ufabc.mensagem.testes;

import java.util.Scanner;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import br.edu.ufabc.mensagem.dao.MensagemDAO;
import br.edu.ufabc.mensagem.modelo.*;

public class MainApplication {

	public static void main(String[] args) {
		
		BlockingQueue<Mensagem> filaMensagens = new ArrayBlockingQueue<Mensagem>(100);
		
		ProdutorPrototype p1 = new ProdutorPrototype(1, filaMensagens);
		//- - - - - - - PROTOTYPE - - - - - - - -
		ProdutorPrototype p2 = p1.clonar();
		ProdutorPrototype p3 = p1.clonar();
		
		ConsumidorSMS c1 = new ConsumidorSMS(filaMensagens);
		//ConsumidorSMS c2 = new ConsumidorSMS(filaMensagens); -> Resulta em erro: padrão SINGLETON - apenas uma instância permitida
		
		p1.start();
		p2.start();
		p3.start();
		
		c1.start();
		
		//- - - - - - - PADR�O DAO (DATA ACCESS OBJECT) - - - - - - - -
		MensagemDAO msgDAO = new MensagemDAO();
		
		//- - - - - - - ITERATOR - - - - - - - -
		for (Mensagem mensagem : filaMensagens) {
			mensagem.statusMensagem();
			msgDAO.insereMensagem(mensagem);
		}
		
		System.out.println("\nDeseja deletar alguma mensagem? [1 = S / 2 = N]");
		Scanner sc = new Scanner(System.in);
		Integer option = Integer.parseInt(sc.next());
		
		if (option == 1){
			msgDAO.deletaMensagem(filaMensagens.element());
		} else {
			System.out.println("Fim do programa.");
		}
		
		
		
		

	}

}
