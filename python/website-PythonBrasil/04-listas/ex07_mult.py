"""4.7 Faça um Programa que leia um vetor de 5 números inteiros, mostre a soma, a multiplicação e os números. """

lista = [int(input('Insira um número inteiro: '))for x in range(5)]
soma = 0
mult = 1

for item in lista:
    soma += item
    mult *= item
print(f'Lista: {lista}\nSoma: {soma}\nMultiplicação: {mult}')