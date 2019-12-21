//	Atividade 01b - Lab - Programação para Web
 
//	Retomada de conceitos de Programação Orientada à Objeto - HERANÇA, ARRAYLIST

package br.edu.ufabc.empresa.testes;

import java.util.ArrayList;
import java.util.Scanner;

import br.edu.ufabc.empresa.modelo.Empresa;
import br.edu.ufabc.empresa.modelo.Funcionario;

public class MainApplication {

	public static void main(String[] args) {
		 
		Empresa empresa1 = new Empresa("Panini Consultoria", "29.690.0001/89"); //Objeto empresa criado
		
		ArrayList<Funcionario> funcs = new ArrayList<Funcionario>(); //ArrayList do tipo Funcionario para armazenar instâncias de Funcionario
		
		funcs.add(new Funcionario("Thiago Panini", "1234567891", true, "Tecnologia", 1500));
		funcs.add(new Funcionario("Antonio Carlos", "987654321", true, "Engenharia", 2500));
		funcs.add(new Funcionario("Trezeguet", "9999000012", true, "Marketing", 1500));
		funcs.add(new Funcionario("Zinedine Zidane", "9898989898", true, "Diretoria", 7500));
		funcs.add(new Funcionario("Ronaldinho Bruxo", "20022006", true, "Química", 5000));
		
		//Adiciona cada funcionário presente no ArrayList
		for (Funcionario func : funcs) {
			empresa1.adicionaFuncionario(func); 
		}
		
		//Lista todos os funcionários
		empresa1.listaFuncionarios();
		
		//Busca funcionário por setor
		empresa1.buscaPorSetor();
		
		//Busca funcionários ativos
		empresa1.buscaFuncionarioAtivo();
		
		//Busca funcionário por CPF
		empresa1.buscaPorCPF();
		
		/*A implementação de um Menu e de um switch case se faz necessária neste exercício,
		 * porém este tipo de implementação servirá de exemplo para as próximas atividades.
		 */

	}

}
