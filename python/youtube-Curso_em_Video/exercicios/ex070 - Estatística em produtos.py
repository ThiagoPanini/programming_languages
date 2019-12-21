"""Desafio 70
Crie um programa que leia o nome e o preço de vários produtos.
O programa deverá perguntar se o usuário vai continuar. No final, mostre:
A) Qual é o total gasto na compra.
B) Quantos produtos custam mais de R% 1.000,00.
C) Qual é o nome do produto mais barato."""

nome = nomebarato = ''
preco = total = mais1000 = precobarato = 0
r = 'S'
i = 1

print('{:_^60}'.format(' LISTA DE COMPRAS - UTENSÍLIOS '))

while r in 'Ss':
    print('-'*10)
    print(f'PRODUTO {i}')
    print('-'*10)
    nome = input('Nome: ').strip().capitalize()
    preco = float(input('Preço: R$'))
    if i == 1 or preco < precobarato:
        nomebarato = nome
        precobarato = preco
    if preco > 1000:
        mais1000 += 1
    total += preco
    r = input('Deseja continuar? [S/N] ')
    while r not in 'SsNn':
        r = input('Deseja continuar? [S/N] ')
    if r not in 'Ss':
        break
    i += 1
print('_'*35)
print(f'Sua lista contém {i} produtos.')
print(f'Total da compra: R${total:.2f}')
print(f'{mais1000} produto(s) custa(m) mais de R$1.000,00')
print(f'{nomebarato} é o produto mais barato e custa R${precobarato:.2f}')

