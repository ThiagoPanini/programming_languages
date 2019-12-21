package br.edu.ufabc.empresa.modelo;

import java.util.ArrayList;
import java.util.Scanner;

public class Empresa {
	
	//Atributos
	private String razaoSocial;
	private String cnpj;
	private ArrayList<Funcionario> funcionarios = new ArrayList<Funcionario>();
	
	//Método Construtor
	public Empresa(String razaoSocial, String cnpj) {
		this.razaoSocial = razaoSocial;
		this.cnpj = cnpj;
	}
	
	//Getters e Setters
	public String getRazaoSocial() {
		return razaoSocial;
	}

	public void setRazaoSocial(String razaoSocial) {
		this.razaoSocial = razaoSocial;
	}

	public String getCnpj() {
		return cnpj;
	}

	public void setCnpj(String cnpj) {
		this.cnpj = cnpj;
	}
//	
//	public void setFuncionario(Funcionario f) {
//		this.funcionarios.add(f);
//	}

	//Adiciona funcionário
	public void adicionaFuncionario(Funcionario func) {
		funcionarios.add(func);
		System.out.println("Funcionário " + func.getNome() + " adicionado com sucesso.");
	}
		
	//Mostra funcionários
	public void listaFuncionarios() {
		System.out.println("\n- LISTA DE FUNCIONÁRIOS DA EMPRESA " + this.getRazaoSocial() + " -");
		for (Funcionario funcionario : funcionarios) {
			funcionario.statusFuncionario();
		}
	}
	
	//Busca por setor
	public void buscaPorSetor() {
		System.out.println("\n- BUSCA FUNCIONÁRIO(S) PELO SETOR -");
		Scanner sc = new Scanner(System.in);
		System.out.print("Digite o setor: ");
		String setor = sc.nextLine();
		int cont = 0;
		for (Funcionario funcionario : funcionarios) {
			if (funcionario.getSetor() == setor.intern()) {
				funcionario.statusFuncionario();
				cont++;
			}
		}
		if(cont==0) {
			System.out.println("Nenhum funcionário encontrado no setor " + setor);
		}
	}
	
	//Lista funcionários ativos
	public void buscaFuncionarioAtivo() {
		System.out.println("\n- BUSCA FUNCIONÁRIO(S) ATIVO(S) -");
		int cont = 0;
		for (Funcionario funcionario : funcionarios) {
			if(funcionario.isStatus()) {
				funcionario.statusFuncionario();
				cont++;
			}
		}
		System.out.println("\nQuantidade de funcionários cadastrados: " + funcionarios.size());
		System.out.println("Quantidade de funcionários ativos: " + cont);
	}
	
	//Busca funcionário por CPF
	public void buscaPorCPF() {
		System.out.println("\n- BUSCA FUNCIONÁRIO POR CPF -");
		Scanner sc = new Scanner(System.in);
		System.out.println("Digite o CPF: ");
		String cpf = sc.nextLine();
		int cont = 0;
		for (Funcionario funcionario : funcionarios) {
			if (funcionario.getCpf() == cpf.intern()) {
				funcionario.statusFuncionario();
				cont ++;
				break;
			}
		}
		if (cont == 0) {
			System.out.println("Nenhum funcionário encontrado com o CPF " + cpf);
		}
	}

	

}
