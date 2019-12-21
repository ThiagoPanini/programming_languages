package br.edu.ufabc.statico.modelo;

public class ConversorTemperatura {
	
	public static double CparaF(double c) {
		System.out.println("\n--- CONVERSÃO C° para F° ---");
		System.out.println(c + "C° equivale em °F a: ");
		return ((9.0f/5.0f) * c) + 32;
	}
	
	public static double CparaK(double c) {
		System.out.println("\n--- CONVERSÃO C° para K° ---");
		System.out.println(c + "C° equivale em K° a: ");
		return (c + 273.15f);
	}
	
	public static double FparaC(double f) {
		System.out.println("\n--- CONVERSÃO F° para C° ---");
		System.out.println(f + "F° equivale em C° a: ");
		return ((5.0f/9.0f) * (f - 32));
	}
	
	public static double FparaK(double f) {
		System.out.println("\n--- CONVERSÃO F° para K° ---");
		System.out.println(f + "F° equivale em K° a: ");
		return ((5.0f/9.0f) * (f + 459.67f));
	}
	
	public static double KparaC(double k) {
		System.out.println("\n--- CONVERSÃO K° para C° ---");
		System.out.println(k + "K° equivale em C° a: ");
		return (k - 273.15);
	}
	
	public static double KparaF(double k) {
		System.out.println("\n--- CONVERSÃO K° para F° ---");
		System.out.println(k + "K° equivale em F° a: ");
		return ((9.0f/5.0f) * k) - 459.67f;
	}

}
