"""4.2 Faça um Programa que leia um vetor de 10 números reais e mostre-os na ordem inversa"""

print(sorted([int(input(f'Digite o {x+1}º elemento da lista: '))for x in range(10)]))