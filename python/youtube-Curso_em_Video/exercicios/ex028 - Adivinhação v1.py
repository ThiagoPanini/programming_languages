#Deafio 28
#Escreve um programa que faça o computador "pensar" em um número inteiro entre 0 e 5
#e peça para o usuário tentr descobrir qual foi o número escolhido pelo computador.

#O programa deverá escrever na tela se o usuário venceu ou perdeu.

from random import randint
from time import sleep #Biblioteca relacionada a tempos e delays

num = randint(0,5) #Faz o computador "Pensar" (Sortear) o número
print('-=-'*20)
print('Vou pensar em um número entre 0 e 5. Tente adivinhar...')
print('-=-'*20)
#print('Teste: num sorteado = {}'.format(num))
user = int(input('Em que número eu pensei?  '))
print('PROCESSANDO...')
sleep(2) #Faz o programa "dormir" por (x) segundos
if user == num:
    print('Você acertou! Eu realmente pensei no número {}. Parabéns!'.format(user))
else:
    print('Boa tentativa, mas na verdade eu pensei no número {}.'.format(num))
print('\nFoi bom jogar com você!')