"""1.16 Faça um programa para uma loja de tintas. O programa deverá pedir o tamanho
em metros quadrados da área a ser pintada. Considere que a cobertura da tinta é
de 1 litro para cada 3 metros quadrados e que a tinta é vendida em latas de 18
litros, que custam R$ 80,00. Informe ao usuário a quantidades de latas de tinta a
serem compradas e o preço total"""

from math import ceil
area = float(input('Insira o valor em m² da área a ser pintada: '))
litros = area/3
latas = litros/18
precolata = 80
print(f'Para pintar uma parede de {area}m² são necessários {ceil(litros):.2f} litros de tinta.')
print(f'Para {ceil(litros):.2f} litros de tinta, é preciso comprar {ceil(latas):.2f} lata(s) de tinta.')
print(f'Valor final da compra: R${ceil(latas)*precolata:.2f}')