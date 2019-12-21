# Desafio 45
# Crie um JOGO que faça o computador jogar Jo-Ken-Po com você.

from random import choice
from time import sleep
import emoji

lista = ['\033[1;30mPEDRA\033[m' + ' ' + emoji.emojize(':fist:', use_aliases=True),
         '\033[1;32mPAPEL\033[m' + ' ' + emoji.emojize(':hand:', use_aliases=True),
         '\033[1;36mTESOURA\033[m' + ' ' + emoji.emojize(':v:', use_aliases=True)]
print('=-'*20)
print('Computador: VAMOS JOGAR JO-KEN-PO?')
print('=-'*20)
user = int(input('Opções:'
                 '\n[ 1 ] para PEDRA.'
                 '\n[ 2 ] para PAPEL.'
                 '\n[ 3 ] para TESOURA.'
                 '\nDigite: '))
jogador = lista[user-1]
computador = choice(lista)
print('JO')
sleep(0.5)
print('KEN')
sleep(0.5)
print('PO!!!')
print('\nJogador: {}'.format(jogador))
print('Computador: {}'.format(computador))
print('=-'*20)
if jogador == lista[0] and computador == lista [1]:
    print('Computador VENCEU! {} ganha de {}!'.format(lista[1], lista[0]))
elif jogador == lista[0] and computador == lista[2]:
    print('Jogador VENCEU! {} ganha de {}!'.format(lista[0], lista[2]))
elif jogador == lista[1] and computador == lista[0]:
    print('Jogador VENCEU! {} ganha de {}!'.format(lista[1], lista[0]))
elif jogador == lista [1] and computador == lista[2]:
    print('Computador VENCEU! {} ganha de {}!'.format(lista[2], lista[1]))
elif jogador == lista[2] and computador == lista[0]:
    print('Computador VENCEU! {} ganha de {}!'.format(lista[0], lista[2]))
elif jogador == lista [2] and computador == lista[1]:
    print('Jogador VENCEU!{} ganha de {}!'.format(lista[2], lista[1]))
else:
    print('EMPATE!')
print('=-'*20)

