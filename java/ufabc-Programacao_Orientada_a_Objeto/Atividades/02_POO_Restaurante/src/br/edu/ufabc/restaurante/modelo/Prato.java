package br.edu.ufabc.restaurante.modelo;

public class Prato {
	
	//Atributos
	private Integer cod_prato;
	private String descricao;
	private Float v_unitario;
	
	//Método Construtor
	public Prato(Integer cod, String desc, Float v){
		this.cod_prato = cod;
		this.descricao = desc;
		this.v_unitario = v;
	}
	
	//Métodos Especiais (Getters e Setters)
	public Integer getCod_prato() {
		return cod_prato;
	}

	public void setCod_prato(Integer cod_prato) {
		this.cod_prato = cod_prato;
	}

	public String getDescricao() {
		return descricao;
	}

	public void setDescricao(String descricao) {
		this.descricao = descricao;
	}

	public Float getV_unitario() {
		return v_unitario;
	}

	public void setV_unitario(Float v_unitario) {
		this.v_unitario = v_unitario;
	}
	
	//Métodos da Classe Prato
	public void statusPrato() {
		System.out.println("\n----- Informações do Prato " + this.getCod_prato() + " -----");
		System.out.println("Descrição: " + this.getDescricao());
		System.out.println("Valor Unitário: " + this.getV_unitario());
	}

}
