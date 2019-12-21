"""2.9 Faça um Programa que leia três números e mostre-os em ordem decrescente."""

n1 = int(input('Primeiro número: '))
n2 = int(input('Segundo número: '))
n3 = int(input('Terceiro número: '))

lista = [n1, n2, n3]
lista.sort()
print('Ordem decrescente: {}'.format(lista[::-1]))