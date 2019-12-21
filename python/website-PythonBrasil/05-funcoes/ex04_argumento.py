"""4. Faça um programa, com uma função que necessite de um argumento. A função retorna o valor de caractere ‘P’,
se seu argumento for positivo, e ‘N’, se seu argumento for zero ou negativo."""


arg = lambda num: 'P' if num > 0 else 'N'

n = int(input('Digite um número: '))

print(f'Resultado: {arg(n)}')