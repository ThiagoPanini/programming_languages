"""Exercício 82
Crie um programa que vai ler vários número e colocar em uma lista.
Depois disso, crie duas listas extras que vão conter apenas os valores pares e ímpares digitados, respectivamente.
Ao final, mostre o conteúdo das três listas geradas.
"""

geral = []
pares = []
impares = []

while True:
    num = int(input('Digite um número: '))
    geral.append(num)
    if num % 2 == 0:
        pares.append(num)
    else:
        impares.append(num)
    r = input('Deseja continuar? [S/N]: ').strip().upper()
    while r not in 'SN':
        r = input('Entrada inválida. Deseja continuar? [S/N] ').strip().upper()
    if r in 'N':
        break

print('-'*15)
print('{:^15}'.format('LISTA GERAL'))
print('-'*15)
print(geral)

print('-'*15)
print('{:^15}'.format('NÚMEROS PARES'))
print('-'*15)
print(pares)

print('-'*15)
print('{:^15}'.format('NÚMEROS ÍMPARES'))
print('-'*15)
print(impares)
