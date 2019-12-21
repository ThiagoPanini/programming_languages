"""Exercício 78
Faça um programa que leia 5 valores numéricos e guarde-os em uma lista.
No final, mostre qual foi o maior e o menor valor digitado e suas respectivas
posições na lista.
"""

numeros = []
for i in range(5):
    numeros.append(int(input(f'Digite o {i+1}º valor: ')))

print('-'*20)
print('{:^20}'.format('LISTA'))
print('-'*20)
print(numeros)
print(f'Maior número: {max(numeros)} na posição {numeros.index(max(numeros))}')
print(f'Menor número: {min(numeros)} na posição {numeros.index(min(numeros))}')