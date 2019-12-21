"""2.7 Faça um Programa que leia três números e mostre o maior e o menor deles."""

n1 = int(input('Primeiro número: '))
n2 = int(input('Segundo número: '))
n3 = int(input('Terceiro número: '))

if n1 > n2 > n3 or n1 > n3 > n2:
    maior = n1
    if n2 > n3:
        menor = n3
    else:
        menor = n2
elif n2 > n1 > n3 or n2 > n3 > n1:
    maior = n2
    if n1 > n3:
        menor = n3
    else:
        menor = n1
elif n3 > n1 > n2 or n3 > n2 > n1:
    maior = n3
    if n1 > n2:
        menor = n2
    else:
        menor = n1
print(f'\nO maior número é {maior}.')
print(f'O menor número é {menor}.')