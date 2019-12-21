# Desafio 58
# Melhore o jogo do DESAFIO 028, onde o computador vai "pensar em um número de 0 a 10.
# Só que agora o jogador vai tentar adivinhar até acertar, mostrando no final quantos palpites foram necessários para vencer.

from random import randint
import emoji

print('='*21)
print('\033[1;30mJOGO DA ADIVINHAÇÃO!\033[m')
print('{:7} X {:6}{}'.format(emoji.emojize(':computer:', use_aliases=True),'', (emoji.emojize(':boy:', use_aliases=True))))
print('='*21)
pc = randint(0, 10)
print(emoji.emojize(':computer:',use_aliases=True),': Pensei em um número de 1 a 10. Tente adivinhar!')
n = int(input(emoji.emojize(':boy:', use_aliases=True)+' : '))
cont = 0

while n != pc: #ou então 'while not acertou: ' (iniciando a variável acertou = False antes
    if n > pc:
        print(emoji.emojize(':computer:', use_aliases=True), ': Menos... Tente novamente.')
        n = int(input(emoji.emojize(':boy:', use_aliases=True) + ' : '))
    if n < pc:
        print(emoji.emojize(':computer:', use_aliases=True), ': Mais... Tente novamente.')
        n = int(input(emoji.emojize(':boy:', use_aliases=True) + ' : '))
    cont += 1
print(emoji.emojize(':computer:',use_aliases=True),': Você ACERTOU com {} tentativas! Parabéns!'.format(cont+1))
#print(emoji.emojize(':computer:',use_aliases=True),': Para ganhar você precisou de {} tentativas! Até a próxima!'.format(cont))