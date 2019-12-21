"""3.23 Faça um programa que mostre todos os primos entre 1 e N sendo N um número inteiro fornecido pelo usuário.
O programa deverá mostrar também o número de divisões que ele executou para encontrar os números primos.
Serão avaliados o funcionamento, o estilo e o número de testes (divisões) executados. """

cont = total = 0
intervalo = []
primo = False

n1 = int(input('Primeiro número do intervalo: '))
n2 = int(input('Segundo número do intervalo: '))
print('Intervalo selecionado: ', end='')

for item in range(n1, n2 + 1):
    if item == n2:
        print(item, end='.')
    else:
        print(item, end=', ')
print('\n')
for c in range(n1, n2 + 1):
    for i in range(1, c + 1):
        if c % i == 0:
            cont += 1
    if cont > 2:
        cont = 0
    else:
        print(f'O número {c} é primo.')
        total += 1
print(f'Entre {n1} e {n2} há um total de {total} números primos.')








