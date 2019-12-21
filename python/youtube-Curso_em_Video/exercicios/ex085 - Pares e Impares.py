"""Exercício 85
Crie um programa onde o usupario possa digitar sete valores numérios e cadastre-os em uma lista única
que mantenha separados os valores pares e ímpares. No final, mostre os valores pares e ímpares em ordem crescente.
"""

numeros = [[], []]

for i in range(7):
    n = int(input(f'Digite o {i+1}º número: '))
    if n % 2 == 0:
        numeros[0].append(n)
    else:
        numeros[1].append(n)

numeros[0].sort()
numeros[1].sort()

print(f'Números digitados: {numeros}')
print(f'Pares: {numeros[0]}')
print(f'Ímpares: {numeros[1]}')