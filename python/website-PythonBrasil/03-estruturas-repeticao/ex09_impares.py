"""3.9 Faça um programa que imprima na tela apenas os números ímpares entre 1 e 50."""

print('~'*30)
print('Números ÍMPARES entre 1 e 50')
print('~'*30)

for c in range(1, 51):
    if c % 2 != 0:
        print(c)
print('Fim.')