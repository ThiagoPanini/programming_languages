# Desafio 46
# Faça um programa que mostre na tela uma contagem regressiva para o estouro de fogos de artifício,
# indo de 10 até 0, com uma pausa de 1 segundo entre eles.

from time import sleep
import emoji

print('\n')
print('=-'*12)
print('CONTAGEM REGRESSIVA...')
print('=-'*12)

for c in range(10, 0, -1):
    print(c)
    sleep(1)

print('\n')
print(emoji.emojize(':boom:'*12, use_aliases=True))
print('FOGOS DE ARTIFÍCIO!')
print(emoji.emojize(':boom:'*12, use_aliases=True))