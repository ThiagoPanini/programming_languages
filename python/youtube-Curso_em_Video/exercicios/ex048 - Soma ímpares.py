# Desafio 48
# Faça um programa que calcule a soma entre todos os números ímpares que são múltiplos de três
# e que se encontram no intervalo de 1 até 500.

print('Entre 1 e 500, temos os seguintes números ÍMPARES e MÚLTIPLOS DE 3:')
s = 0
for c in range(1, 501, 2):
    if c % 3 == 0:
        print(c)
        s = s + c
print('A soma entre eles é igual a {}'.format(s))
print('Fim')