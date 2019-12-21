"""Exercício 81
Crie um programa que vai ler vários números e colocar em uma lista. Depois disso, mostre:
A) Quantos números foram digitados.
B) A lita de valores ordenada de forma decrescente.
C) Se o valor 5 foi digitado e está ou não na lista.
"""

numeros = []
cont = 0
while True:
    numeros.append(int(input('Digite um número: ')))
    cont += 1
    r = input('Deseja continuar? [S/N] ').strip().upper()
    if r in 'N':
        break

print('-'*10)
print('{:^10}'.format('ANÁLISE'))
print('-'*10)
print(f'Foram digitados {cont} números.')
numeros.sort(reverse=True)
print(f'Lista ordenada de forma decrescente: {numeros}')
if 5 in numeros:
    print('O número 5 está na lista.')
else:
    print('O número 5 não está na lista.')