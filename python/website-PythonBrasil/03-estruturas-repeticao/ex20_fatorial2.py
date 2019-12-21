"""3.20 Altere o programa de cálculo do fatorial, permitindo ao usuário calcular o fatorial várias vezes e limitando o
fatorial a números inteiros positivos e menores que 16."""

resp = 'Ss'
while resp in 'Ss':
    n = int(input('Digite um número para ver seu FATORIAL [entre 0 e 16]: '))
    while n not in range(17):
        n = int(input('Número inválido. Digite um inteiro entre 0 e 16: '))
    r = n
    print(f'{n}', end='')

    for c in range(n - 1):
        print(f' x {n-1}', end='')
        n -= 1
        r *= n
    print(f' = {r}.')
    resp = input('Deseja continuar? [S/N] ').strip().upper()[0]
print('Fim.')