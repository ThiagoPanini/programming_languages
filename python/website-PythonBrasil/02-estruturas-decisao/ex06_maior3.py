"""2.6 Faça um Programa que leia três números e mostre o maior deles."""

n1 = int(input('Primeiro número: '))
n2 = int(input('Segundo número: '))
n3 = int(input('Terceiro número: '))

if n1 > n2 > n3 or n1 > n3 > n2:
    maior = n1
elif n2 > n1 > n3 or n2 > n3 > n1:
    maior = n2
elif n3 > n1 > n2 or n3 > n2 > n1:
    maior = n3

print(f'O maior número é {maior}.')