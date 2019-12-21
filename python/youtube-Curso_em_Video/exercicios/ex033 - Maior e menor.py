#Deafio 33
#Faça um programa que leia três números e mostre qual é o MAIOR e qual é o MENOR

n1 = int(input('Digite o primeiro número inteiro: '))
n2 = int(input('Digite o segundo número inteiro: '))
n3 = int(input('Digite o terceiro número inteiro: '))

#______LÓGICA PENSADA INICIALMENTE________
"""if n1 > n2 > n3:
    print('O maior número é {}.'.format(n1))
    print('O menor número é {}.'.format(n3))
if n1 > n3 > n2:
    print('O maior número é {}.'.format(n1))
    print('O menor número é {}.'.format(n2))
if n2 > n1 > n3:
    print('O maior número é {}.'.format(n2))
    print('O menor número é {}.'.format(n3))
if n2 > n3 > n1:
    print('O maior número é {}.'.format(n2))
    print('O menor número é {}.'.format(n1))
if n3 > n1 > n2:
    print('O maior número é {}.'.format(n3))
    print('O menor número é {}.'.format(n2))
if n3 > n2 > n1:
    print('O maior número é {}.'.format(n3))
    print('O menor número é {}.'.format(n1))"""

#___________LÓGICA VÍDEO RESOLUÇÃO_____________
menor = n1 #Já considera logo de cara que o menor é o primeiro número. Se estiver OK, eliminamos um passo
if n2<n1 and n2<n3:
    menor = n2 #Se o segundo número for menor que o primeiro E menor que o terceiro, o menor então é o segundo
if n3<n1 and n3<n2:
    menor = n3 #Se o terceiro número for menor que o primeiro E menor que o segundo, o menor então é o terceiro
print('O menor valor digitado foi {}'.format(menor))
#Repetir a verificação para o maior
maior = n1
if (n2 > n1) and (n3 > n1):
    maior = n2
if n3>n1 and n3>n2:
    maior = n3
print('O maior valor digitado foi {}'.format(maior))
