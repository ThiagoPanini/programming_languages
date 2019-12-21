# Desafio 50
# Desenvolva um programa que leia seis números inteiros e mostre a soma apenas daqueles que forem pares.
# Se o valor digitado for ímpar, desconsidere-o.

s = 0
for c in range(1, 7):
    n = int(input('Insira o {}° número entre seis: '.format(c)))
    if n % 2 == 0:
        print('\033[32mNúmero PAR digitado. Somando.\033[m')
        s = s + n
    else:
        print('\033[1;31mNúmero ÍMPAR digitado. Desconsiderado.\033[m')
    print('Soma atualizada: \033[1;30m{}\033[m'.format(s))
print('\n\033[1;30mSoma final: {}\033[m'.format(s))
print('Fim.')