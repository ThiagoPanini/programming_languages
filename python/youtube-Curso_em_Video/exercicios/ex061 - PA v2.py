# Desafio 61
# Refraça o DESAIO 051, lendo o primeiro termo e a razão de uma Pa, mostranso os 10 primeiros termos usnado while.

print('='*30)
print('PRIMEIROS 10 TERMOS DE UMA PA')
print('='*30)

a1 = int(input('Primeiro termo: '))
a1x = a1
r = int(input('Razão: '))
rx = r
i = 1
print('='*30)
while i <= 10:
    print('Termo a{} da PA é igual a {}'.format(i, a1))
    a1 = a1 + r
    i += 1
print('='*35)
print('COLOCANDO OS VALORES EM LINHA ÚNICA')
print('='*35)
i = 1
while i <= 10:
    print('{}'.format(a1x), end='')
    print('→' if i < 10 else '', end='')
    a1x += rx
    i += 1
print('\nFim')
