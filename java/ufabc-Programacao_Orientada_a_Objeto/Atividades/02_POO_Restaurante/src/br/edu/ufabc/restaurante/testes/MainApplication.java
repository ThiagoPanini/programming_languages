/*Link da atividade:
 * http://tidia4.ufabc.edu.br/portal/site/45fa0b60-2315-4dfe-8e62-5748bc4ad8ba/page/aa181a58-4eca-4389-8baf-e9792fcbf796
 */

package br.edu.ufabc.restaurante.testes;

import java.util.Random;

import br.edu.ufabc.restaurante.modelo.Garcom;
import br.edu.ufabc.restaurante.modelo.Mesa;
import br.edu.ufabc.restaurante.modelo.Pedido;
import br.edu.ufabc.restaurante.modelo.Prato;

import java.util.ArrayList;

public class MainApplication {

	public static void main(String[] args) {
		
		Random aleatorio = new Random();
		
		Mesa m1 = new Mesa(1, 4, aleatorio.nextInt(5));
		Mesa m2 = new Mesa(2, 4, aleatorio.nextInt(5));
		Mesa m3 = new Mesa(3, 4, aleatorio.nextInt(5));
		Mesa m4 = new Mesa(4, 4, aleatorio.nextInt(5));
		Mesa m5 = new Mesa(5, 4, aleatorio.nextInt(5));
		//m1.statusMesa();
		
		Garcom g1 = new Garcom(101, "Campeão");
		Garcom g2 = new Garcom(102, "Mestre");
		Garcom g3 = new Garcom(103, "Tobias");
		Garcom g4 = new Garcom(104, "Valter");
		Garcom g5 = new Garcom(105, "Garrafinha");
		//g1.statusGarcom();
		
		ArrayList<Prato> pratos = new ArrayList<Prato>();
		Prato p1 = new Prato(101, "Filé de Frango", 15.50f);
		pratos.add(p1);
		Prato p2 = new Prato(102, "Bisteca Suína", 25.50f);
		pratos.add(p2);
		Prato p3 = new Prato(103, "Fritas com Bacon", 10.00f);
		pratos.add(p3);
		Prato p4 = new Prato(104, "Feijão de Corda", 8.50f);
		pratos.add(p4);
		Prato p5 = new Prato(105, "Arroz Branco", 8.00f);	
		pratos.add(p5);
		//p1.statusPrato();
		

		Pedido ped1 = new Pedido(101, m1, g1);
		ped1.adicionaPrato(p1);
		ped1.adicionaPrato(p4);
		ped1.apresentaPedido();
		
		Pedido ped2 = new Pedido(102, m2, g4);
		ped2.adicionaPrato(p3);
		ped2.adicionaPrato(p2);
		ped2.adicionaPrato(p5);
		ped2.apresentaPedido();
		
		Pedido ped3 = new Pedido(103, m5, g2);
		ped3.adicionaPrato(p2);
		ped3.adicionaPrato(p4);
		ped3.adicionaPrato(p4);
		ped3.apresentaPedido();
		
		Pedido ped4 = new Pedido(104, m4, g3);
		ped4.adicionaPrato(p3);
		ped4.adicionaPrato(p2);
		ped4.adicionaPrato(p1);
		ped4.apresentaPedido();
		
		Pedido ped5 = new Pedido(105, m3, g5);
		ped5.adicionaPrato(p5);
		ped5.apresentaPedido();
		
		
	}

}
