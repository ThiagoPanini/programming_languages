# Desafio 51
# Desenvolva um programa que leia o primeiro termpo e a razão de uma PA.
# No final, mostre os 10 primeiros termos dessa progressão.

print('='*30)
print('PRIMEIROS 10 TERMOS DE UMA PA')
print('='*30)
a1 = int(input('Primeiro termo: '))
r = int(input('Razão: '))
i = 1
print('=-'*10)
for a1 in range(a1, a1+10*r, r): # Fórmula do 11° termo para que o laço possa contar o 10° termo. Caso contrário, ele para no 9°
    print('Termo a{} da PA: {}'.format(i, a1))
    i = i+1
print('=-'*10)
print('\nFim')