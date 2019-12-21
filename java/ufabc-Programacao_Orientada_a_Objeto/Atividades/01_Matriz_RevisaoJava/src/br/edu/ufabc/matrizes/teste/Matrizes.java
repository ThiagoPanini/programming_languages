package br.edu.ufabc.matrizes.teste;

import java.util.Scanner;

public class Matrizes {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        int i, j, k, result;
        
        Scanner sc = new Scanner(System.in);
        System.out.println("MULTIPLICAÇÃO DE MATRIZES\n");
        System.out.println("Entre com o tamanho das matrizes:");
        int size = sc.nextInt();
        int m[][] = new int[size][size];
        int n[][] = new int[size][size];
        int mn[][] = new int[size][size];
        
        // Preenchendo as matrizes
        System.out.println("MATRIZ 'M'");
        for (i = 0; i < size; i++){
            System.out.println("Informe os elementos da "+(i+1)+"ª linha: ");
            for (j = 0; j < size; j++){
                System.out.printf("Matriz[%d][%d] = ", i+1, j+1);
                m[i][j] = sc.nextInt();
            }
            System.out.printf("\n");
        }
        
        System.out.println("MATRIZ 'N'");
        for (i = 0; i < size; i++){
            System.out.println("Informe os elementos da "+(i+1)+"ª linha: ");
            for (j = 0; j < size; j++){
                System.out.printf("Matriz[%d][%d] = ", i+1, j+1);
                n[i][j] = sc.nextInt(); // Mudar no método
            }
            System.out.printf("\n");
        }
        
        // Multiplicando as matrizes
        for (i=0; i<size; i++) {
            for (j=0; j<size; j++) {
                result = 0;
                for (k=0; k<size; k++) {
                    result = result + (m[i][k] * n[k][j]);
                }
                mn[i][j] = result;
            }
        }
        
        // Imprimindo as matrizes
        System.out.println("RESULTADO M x N");
        for (i=0; i<size; i++) {
            for (j=0; j<size; j++) {
                if (j == 0){
                    System.out.print("[");
                    System.out.print(mn[i][j]);
                }
                else{
                    if (j != 0){
                        System.out.print(", " + mn[i][j]);
                    }
                    else{
                        System.out.print(mn[i][j]);
                    }
                }
                if (j == size - 1){
                    System.out.print("]");
                }
            }
            System.out.printf("\n");
        }
    }
    
  //Ideias para o futuro:
	
    //Atributos
    //size (tamanho das matrizes)
    
    //Métodos        
    //preencheMatriz(m, size)
    //preencheMatriz(n, size)
    //multiplicaMatriz(m, n, mn)
    //imprimeMatriz("Matriz 1", m)
    //imprimeMatriz("Matriz 2", n)
    //imprimeMatriz("Resultado", mn)

}
