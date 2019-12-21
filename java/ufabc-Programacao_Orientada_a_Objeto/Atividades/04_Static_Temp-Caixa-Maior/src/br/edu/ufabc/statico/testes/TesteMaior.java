package br.edu.ufabc.statico.testes;

import java.util.Scanner;

import br.edu.ufabc.statico.modelo.MaiorNumero;

public class TesteMaior {

	public static void main(String[] args) {
		
		Double n1, n2, n3, n4;
		
		Scanner sc = new Scanner(System.in);
				
		System.out.println("O que deseja verificar?\n"
				+ "[ 1 ] para maior entre dois números\n"
				+ "[ 2 ] para maior entre três números\n"
				+ "[ 3 ] para maior entre quatro números\n");
		
		Integer option = Integer.parseInt(sc.nextLine());
		
		System.out.println("Insira o primeiro número: ");
		n1 = Double.parseDouble(sc.nextLine());
		System.out.println("Insira o segundo número: ");
		n2 = Double.parseDouble(sc.nextLine());
		
		
		switch(option) {
		case 1:
			double maior2 = MaiorNumero.maiorDeDois(n1, n2);		
			System.out.println("O maior entre estes dois números é: " + maior2);
			break;
			
		case 2:
			System.out.println("Insira o terceiro número: ");
			n3 = Double.parseDouble(sc.nextLine());
			double maior3 = MaiorNumero.maiorDeTres(MaiorNumero.maiorDeDois(n1, n2), n3);
			System.out.println("O maior entre estes três números é: " + maior3);
			break;
			
		case 3:
			System.out.println("Insira o terceiro número: ");
			n3 = Double.parseDouble(sc.nextLine());
			System.out.println("Insira o quarto número: ");
			n4 = Double.parseDouble(sc.nextLine());
			double maior4 = MaiorNumero.maiorDeQuatro(MaiorNumero.maiorDeTres(MaiorNumero.maiorDeDois(n1, n2), n3), n4);
			System.out.println("O maior entre estes quatro números é: " + maior4);
			break;
		
		default: System.out.println("Entrada inválida.");			
		}
		
		
	}

}
