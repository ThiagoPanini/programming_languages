"""2.2 Faça um Programa que peça um valor e mostre na tela se o valor é positivo ou negativo. """

n1 = float(input('Insira um valor: '))

if n1 > 0:
    print(f'{n1} é positivo.')
elif n1 < 0:
    print(f'{n1} é negativo.')
else:
    print('Digitado 0 ou um valor inválido.')