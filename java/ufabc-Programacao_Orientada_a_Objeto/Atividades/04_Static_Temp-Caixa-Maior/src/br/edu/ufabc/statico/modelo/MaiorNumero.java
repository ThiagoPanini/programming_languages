package br.edu.ufabc.statico.modelo;

public class MaiorNumero {
	
	public static double maiorDeDois(double n1, double n2) {
		
		if(n1>n2) {
			return n1;
		} else if(n2>n1) {
			return n2;
		} else {
			System.out.println("Números iguais");
			return n1;
		}
	}
	
	public static double maiorDeTres(double maior2, double n3) {
		
		if(n3>maior2) {
			return n3;
		} else {
			return maior2;
		}
	}
	
	public static double maiorDeQuatro(double maior3, double n4) {
		
		if(n4>maior3) {
			return n4;
		} else {
			return maior3;
		}
	}

}
