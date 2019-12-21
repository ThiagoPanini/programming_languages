"""Exercício 88
Faça um programa que ajude um jogador da MEGA SENA a criar palpites. O programa vai perguntar quantos jogos serão
gerados e vai sortear 6 números entre 1 e 60 para cada jogo, cadastrando tudo em uma lista composta.
"""

from time import sleep
from random import randint

loterica = []
jogos = []

print('-' * 40)
print(f'{"JOGOS DA MEGA SENA":^40}')
print('-' * 40)

n = int(input('Quantos jogos deseja fazer? '))
sleep(1)

for i in range(n):
    while len(jogos) < 6:
        num_sorteado = randint(1, 60)
        if num_sorteado in jogos:
            continue
        else:
            jogos.append(num_sorteado)
    loterica.append(jogos[:])
    jogos.clear()
    print(f'Jogo número {i+1}: {loterica[i]}')
    sleep(1)

print('-' * 40)
print(f'{"BOA SORTE":^40}')
print('-' * 40)


