"""Exercício 86
Crie um programa que crie uma matriz de dimensão 3x3 e preencha com valores lidos pelo teclado.
No final, mostre a matriz na tela, com a formatação correta.
"""
matriz = [[], [], []]

for i in range(3):
    for j in range(3):
        matriz[i].append(input(f'Digite um valor para o elemento [{i+1}][{j+1}]: '))

# Imprimindo matriz
print('-'*10)
for i in range(3):
    for j in range(3):
        print(f'[{matriz[i][j]:^5}]', end='')
    print()
print('-'*10)


