//	Atividade 01a - Lab - Programação para Web
 
//	Retomada de conceitos de Programação Orientada à Objeto

package br.com.bb.sistemabancario.testes;

import br.com.bb.sistemabancario.modelo.ContaCorrente;

import java.util.Random;
import java.util.Scanner;

public class MainContaCorrente {

	public static void main(String[] args) {
				
		Scanner sc = new Scanner(System.in);
		Random rd = new Random();
		System.out.println("Insira a quantidade de contas a serem cadastradas: ");
		Integer qtd = Integer.parseInt(sc.next());
		
		ContaCorrente contas[] = ContaCorrente.adicionaConta(qtd);
		
		for (ContaCorrente conta : contas){
			conta.statusConta();
		}
		
		
	
	}
}