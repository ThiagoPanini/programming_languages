package br.edu.ufabc.empresa.modelo;

public class Funcionario {
	
	//Atributos
	private Integer id;
	private static Integer cont = 1;
	private String nome;
	private String cpf;
	private boolean status;
	private String setor;
	private float salarioMensal;
	
	//Método Construtor
	public Funcionario(String nome, String cpf, boolean status, String setor, float salarioMensal) {
		this.id = cont;
		cont++;
		this.nome = nome;
		this.cpf = cpf;
		this.status = status;
		this.setor = setor;
		this.salarioMensal = salarioMensal;
	}
	
	//Getters e Setters
	public Integer getId() {
		return id;
	}
	
	public void setId(Integer id) {
		this.id = id;
	}
	
	public String getNome() {
		return nome;
	}
	
	public void setNome(String nome) {
		this.nome = nome;
	}

	public String getCpf() {
		return cpf;
	}

	public void setCpf(String cpf) {
		this.cpf = cpf;
	}

	public boolean isStatus() {
		return status;
	}

	public void setStatus(boolean status) {
		this.status = status;
	}

	public String getSetor() {
		return setor;
	}

	public void setSetor(String setor) {
		this.setor = setor;
	}

	public float getSalarioMensal() {
		return salarioMensal;
	}

	public void setSalarioMensal(float salarioMensal) {
		this.salarioMensal = salarioMensal;
	}

	//Método para calcular o salário anual do funcionário
	public float salarioAnual() {
		return this.getSalarioMensal()*12;
	}
	
	//Método para mostrar status do funcionário
	public void statusFuncionario() {
		System.out.println("\nId: " + this.getId());
		System.out.println("Nome: " + this.getNome());
		System.out.println("CPF: " + this.getCpf());
		System.out.println("Setor: " + this.getSetor());
		System.out.println("Status: " + this.isStatus());
		System.out.println("Salário Mensal: R$" + this.getSalarioMensal());
		System.out.println("Salário Anual: R$" + this.salarioAnual());
	}
	

}
