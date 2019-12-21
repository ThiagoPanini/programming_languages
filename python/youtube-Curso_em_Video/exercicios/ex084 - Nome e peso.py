"""Exercício 84
Faça um programa que leia nome e peso de várias pessoas, guardando tudo em uma lista.
No final, mostre:
A) Quantas pessoas foram cadastradas
B) Uma listagem com as pessoas mais pesadas
C) Uma listagem com as pessoas mais leves"""

pessoas = []
dados = []
cont = maior = menor = 0

while True:
    print('{:-^30} '.format('Cadastro da pessoa ' + str(cont+1)))
    dados.append(input('Nome: '))
    dados.append(float(input('Peso [Kg]: ')))
    if len(pessoas) == 0:
        maior = menor = dados[1]
    else:
        if dados[1] > maior:
            maior = dados[1]
        if dados[1] < menor:
            menor = dados[1]
    pessoas.append(dados[:])
    dados.clear()
    r = input('Deseja continuar? [S/N] ').strip().upper()
    cont += 1
    while r not in 'SN':
        r = input('Entrada inválida. Deseja continuar? [S/N] ').strip().upper()
    if r in 'N':
        break

print(pessoas)
print('-'*40)
print(f'Você cadastrou {len(pessoas)} pessoas.')
print(f'O maior peso foi de {maior}Kg. Peso de ', end='')
for p in pessoas:
    if p[1] == maior:
        print(f'[{p[0]}]', end=' ')
print()
print(f'O menor peso foi de {menor}Kg. Peso de ', end='')
for p in pessoas:
    if p[1] == menor:
        print(f'[{p[0]}]', end=' ')


