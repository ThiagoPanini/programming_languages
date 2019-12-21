"""3.11 Altere o programa anterior para mostrar no final a soma dos números. """

print('-'*25)
print('Soma de um Intervalo')
print('-'*25)

n1 = int(input('Primeiro número '))
n2 = int(input('Segundo número: '))

soma = 0

for c in range(n1+1, n2):
    soma += c
print(f'A soma dos inteiros entre {n1} e {n2} é igual a {soma}.')