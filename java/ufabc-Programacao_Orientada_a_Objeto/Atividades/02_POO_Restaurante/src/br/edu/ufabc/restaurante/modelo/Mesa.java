package br.edu.ufabc.restaurante.modelo;

public class Mesa {
	
	//Atributos
	private Integer num_mesa, lugares, ocupantes;
	private Boolean disponivel, escolha, encerra;
	
	//Método Construtor
	public Mesa(Integer n, Integer l, Integer o) {
		this.num_mesa = n;
		this.lugares = l;
		this.ocupantes = o;
		this.disponivel = false;
		this.escolha = false;
		this.encerra = false;
	}


	//Métodos Especiais (Getters e Setters)
	public Integer getNum_mesa() {
		return num_mesa;
	}
	
	public void setNum_mesa(Integer num_mesa) {
		this.num_mesa = num_mesa;
	}
	
	public Integer getLugares() {
		return lugares;
	}
	
	public void setLugares(Integer lugares) {
		this.lugares = lugares;
	}
	
	public Integer getOcupantes() {
		return ocupantes;
	}
	
	public void setOcupantes(Integer ocup) {
		this.ocupantes = ocup;
	}
	
	public Boolean getDisponivel() {
		return disponivel;
	}
	
	public void setDisponivel(Boolean disponivel) {
		this.disponivel = disponivel;
	}
	
	public Boolean getEscolha() {
		return escolha;
	}
	
	public void setEscolha(Boolean escolha) {
		this.escolha = escolha;
	}
	
	public Boolean getEncerra() {
		return encerra;
	}
	
	public void setEncerra(Boolean encerra) {
		this.encerra = encerra;
	}
	
	//Métodos da Classe Mesa
	public void statusMesa() {
		System.out.println("\n----- Status da Mesa " + this.getNum_mesa() + " -----");
		System.out.println("Dispon�vel? " + this.getDisponivel());
		System.out.println("Lugares: " + this.getLugares());
		
		if (!this.getDisponivel()) {
			System.out.println("Ocupantes: " + this.getOcupantes());
		}		
	}
	 
	public void chamarGarcom() {
		this.setEscolha(true);
	}

}
