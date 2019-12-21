package br.edu.ufabc.statico.testes;

import br.edu.ufabc.statico.modelo.ConversorTemperatura;

public class TesteConversao {

	public static void main(String[] args) {
		
		System.out.println(ConversorTemperatura.CparaF(100));
		System.out.println(ConversorTemperatura.CparaK(100));
		System.out.println(ConversorTemperatura.FparaC(150));
		System.out.println(ConversorTemperatura.FparaK(150));
		System.out.println(ConversorTemperatura.KparaC(250));
		System.out.println(ConversorTemperatura.KparaF(250));

	}

}
