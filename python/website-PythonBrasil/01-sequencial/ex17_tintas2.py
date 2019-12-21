"""1.17 Faça um Programa para uma loja de tintas. O programa deverá pedir o tamanho
em metros quadrados da área a ser pintada. Considere que a cobertura da tinta é
de 1 litro para cada 6 metros quadrados e que a tinta é vendida em latas de 18
litros, que custam R$ 80,00 ou em galões de 3,6 litros, que custam R$ 25,00.
 Informe ao usuário as quantidades de tinta a serem compradas e os
respectivos preços em 3 situações:
 a. comprar apenas latas de 18 litros;
 b. comprar apenas galões de 3,6 litros;
 c. misturar latas e galões, de forma que o preço seja o menor.
 Acrescente 10% de folga e sempre arredonde os valores para cima,
isto é, considere latas cheias."""

from math import ceil

print('~'*22)
print('LOJA DE TINTAS PYTHON!')
print('~'*22)
area = float(input('Área em m² a ser pintada: '))
litros = area/6
lata = 18
qtdlatas = ceil(litros/18)
precolata = 80
galao = 3.6
qtdgalao = ceil(litros/3.6)
precogalao = 25

print(f'Para pintar uma parede de {area}m² são necessários {litros:.2f} litro(s) de tinta.')
print(f'\nSituação 1: Será preciso comprar {qtdlatas} lata(s) de 18L ao valor de R${qtdlatas*precolata:.2f}.')
print(f'Sobra: {(qtdlatas*lata)-litros:.2f} litro(s) de tinta.')
print(f'\nSituação 2: Será preciso comprar {qtdgalao} galão(ões) de 3.6L ao valor de R${qtdgalao*precogalao:.2f}.')
print(f'Sobra: {(qtdgalao*galao)-litros:.2f} litro(s) de tinta.')

qtdlatas -= 1
sobra = litros-(qtdlatas*lata)
qtdgalao = ceil(sobra/3.6)
print(f'\nSituação 3: Compre {qtdlatas} lata(s) de tinta + {qtdgalao} galão(ões) a um valor total de '
      f'R${(qtdlatas*precolata)+(qtdgalao*precogalao):.2f}')
print(f'Sobra: {(qtdgalao*galao)-sobra:.2f} litro(s) de tinta.')

