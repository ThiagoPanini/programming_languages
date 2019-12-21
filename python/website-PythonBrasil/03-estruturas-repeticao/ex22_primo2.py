"""3.22 Altere o programa de cálculo dos números primos, informando, caso o número não seja primo,
por quais número ele é divisível."""

# O exercício 21 já foi realizado dessa forma.

r = 'Ss'
primo = False

while r in 'Ss':
    n = int(input('Digite um número inteiro: '))
    metade = int(n/2)
    for c in range(2, metade+1):
        if n % c == 0:
            print(f'O número {n} é divisível por {c}.')
            primo = False
        else:
            primo = True
    if primo:
        print(f'O número {n} é primo.')
    else:
        print(f'O número {n} não é primo.')
    r = input('Deseja continuar? [S/N] ').strip().upper()[0]
print('Fim.')