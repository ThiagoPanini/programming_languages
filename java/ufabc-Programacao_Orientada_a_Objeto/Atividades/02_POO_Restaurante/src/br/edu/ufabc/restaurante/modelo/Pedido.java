package br.edu.ufabc.restaurante.modelo;

import java.util.ArrayList;

public class Pedido {
	
	//Atributos
	private Integer num, qtd;
	private Mesa mesa;
	private Garcom garcom;
	ArrayList<Prato> pratos;
	
	//Método Construtor
	public Pedido(Integer n, Mesa m, Garcom g) {
		this.mesa = m;
		this.garcom = g;
		this.num = n;
		pratos = new ArrayList<Prato>();
	}
	
	//Métodos Getters e Setters
	public Integer getNum() {
		return num;
	}

	public void setNum(Integer num) {
		this.num = num;
	}

	public Integer getMesa() {
		return mesa.getNum_mesa();
	}

	public String getGarcom() {
		return garcom.getNome();
	}
	
	public void adicionaPrato(Prato prato){
		pratos.add(prato);
	}

	//Método Menu	
	public void getPrato(){
		for (Prato p : pratos) {
			System.out.println("Descrição: " + p.getDescricao() + " - R$ " + p.getV_unitario());
		}
	}
	
	//Métodos da Classe Pedido	
	public void apresentaPedido(){
		System.out.println("\n------ Pedido " + this.getNum() + " ------");
		System.out.println("Mesa atendida: " + this.getMesa());
		System.out.println("Garçom responsável: " + this.getGarcom());
		System.out.println("-------- Pratos --------");
		for (Prato p : pratos) {
			System.out.println(p.getDescricao() + " - R$ " + p.getV_unitario());
		}
	}
}
