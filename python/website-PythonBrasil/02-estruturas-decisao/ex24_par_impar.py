"""2.24 Faça um Programa que peça um número inteiro e determine se ele é par ou impar.
Dica: utilize o operador módulo (resto da divisão). """

n = int(input('Digite um número inteiro: '))
resultado = ''
if n % 2 == 0:
    resultado = 'PAR'
else:
    resultado = 'ÍMPAR'
print(f'O número {n} é {resultado}.')