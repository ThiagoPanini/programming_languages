"""3.17 Faça um programa que calcule o fatorial de um número inteiro fornecido pelo usuário. Ex.: 5!=5.4.3.2.1=120"""

n = int(input('Digite um número para ver seu FATORIAL: '))
r = n
print(f'{n}', end='')

for c in range(n-1):
    print(f' x {n-1}', end='')
    n -= 1
    r *= n
print(f' = {r}.\n')

