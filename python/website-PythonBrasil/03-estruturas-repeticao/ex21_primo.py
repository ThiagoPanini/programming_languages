"""3.21 Faça um programa que peça um número inteiro e determine se ele é ou não um número primo.
Um número primo é aquele que é divisível somente por ele mesmo e por 1. """

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