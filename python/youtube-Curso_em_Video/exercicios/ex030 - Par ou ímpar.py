#Deafio 3
#Crie um programa que leia um número inteiro e mostre na tela se ele é PAR ou ÍMPAR

num = int(input('Digite um número inteiro: '))

if num%2 == 0: #Verifica se o resto da divisão por 2 é 0 (Se o número é PAR, é divisível por 2)
    print('O número {} é PAR.'.format(num))
else:
    print('O número {} é IMPAR.'.format(num))