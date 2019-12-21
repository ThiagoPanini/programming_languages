"""3.10 Faça um programa que receba dois números inteiros e gere os números inteiros que estão no
intervalo compreendido por eles. """

print('_'*21)
print('Intervalo de Inteiros')
print('*'*21)

n1 = int(input('Primeiro número inteiro: '))
n2 = int(input('Segundo número inteiro: '))

print(f'Inteiros entre {n1} e {n2}: ', end='')
for c in range(n1+1, n2):
    if c == n2-1:
        print(c, end='.')
    elif c == n2-2:
        print(c, end=' e ')
    else:
        print(c, end=', ')