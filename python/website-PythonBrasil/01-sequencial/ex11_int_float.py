"""1.11 Faça um Programa que peça 2 números inteiros e um número real. Calcule e
mostre:
 a. o produto do dobro do primeiro com metade do segundo .
b. a soma do triplo do primeiro com o terceiro.
 c. o terceiro elevado ao cubo.
"""

int1 = int(input('Digite um número INTEIRO: '))
int2 = int(input('Digite outro número INTEIRO: '))
real = float(input('Digite um número REAL: '))

print(f'O produto do dobro do primeiro com metade do segundo: {(2*int1)*(int2/2)}.')
print(f'A soma do triplo do primeiro com o terceiro: {(3*int1)+real}.')
print(f'O terceiro elvado ao cubo: {real*real*real:.2f}')
