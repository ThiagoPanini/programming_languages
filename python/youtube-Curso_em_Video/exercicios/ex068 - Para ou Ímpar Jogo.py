# Desafio 68
# Faça um programa que jogue par ou ímpar com o computador.
# O jogo só será interrompido quando o jogador PERDER, mostrando o total de vitórias consecutivas que ele conquistou
# no final do jogo.

from time import sleep
from random import randint
import emoji

cont = 0
escolhapc = ''
lista = [emoji.emojize(':boy:', use_aliases=True), emoji.emojize(':computer:', use_aliases=True)]
opcoes = ['PAR', 'ÍMPAR']
# Lista[0] = Jogador
# Lista[1] = Computador

print('~'*30)
print(lista[1] + ': VAMOS JOGAR PAR OU ÍMPAR?')
print('~'*30)
print('{:6}{} x {}{}{:5}{}'.format(lista[0],'HUMANO', '', 'MÁQUINA', '', lista[1]))
while True:
    print('-' * 30)
    escolha = str(input((lista[0]+' JOGADOR - Escolha Par ou Ímpar [P/I]: '))).strip().upper()
    while escolha not in 'PpIi':
        escolha = input((lista[0] + ' JOGADOR - Opção inválida, escolha Par ou Ímpar [P/I]: ')).strip().upper()
    numjogador = int(input(lista[0]+' JOGADOR - Jogue (de 0 a 10): '))
    while 0 > numjogador > 10:
        numjogador = int(input(lista[0] + ' JOGADOR - Opção inválida, escolha entre 0 e 10: '))
    numpc = randint(1, 10)
    print('-'*10)
    print('VAMOS LÁ!')
    print('-'*10)
    if escolha in 'Pp':
        escolha = 'PAR'
        escolhapc = 'ÍMPAR'
    elif escolha in 'Ii':
        escolha = 'ÍMPAR'
        escolhapc = 'PAR'
    print(lista[0] + ' JOGADOR: {}'.format(escolha))
    sleep(1)
    print(lista[1] + ' COMPUTADOR: {}'.format(escolhapc))
    sleep(1.5)
    print(lista[0] + ': {}'.format(numjogador))
    print(lista[1] + ': {}'.format(numpc))
    print('~'*30)
    sleep(1.5)
    if (numjogador + numpc) % 2 == 0 and escolha == 'PAR':
        print(lista[0] + ' JOGADOR VENCEU! {} é Par.'.format(numjogador + numpc))
        print('Reiniciando partida...')
        cont += 1
    elif (numjogador + numpc) % 2 != 0 and escolha == 'ÍMPAR':
        print(lista[0] + ' JOGADOR VENCEU! {} é Ímpar.'.format(numjogador + numpc))
        print('Reiniciando partida...')
        cont += 1
    elif (numjogador + numpc) % 2 == 0 and escolha == 'ÍMPAR':
        print(lista[0] + ' JOGADOR PERDEU! {} é Par'.format(numjogador + numpc))
        print('Encerrando partida...')
        break
    else:
        print(lista[0] + ' JOGADOR PERDEU! {} é Ímpar'.format(numjogador + numpc))
        print('Encerrando partida...')
        break
print('\nVocê venceu {} vez(es) contra o Computador! Volte sempre!'.format(cont))