# Desafio 62
# Melhore o DESAFIO 061, perguntando para o usuário se ele quer mostrar mais alguns termos.
# O programa encerra quando ele disser que quer mostrar 0 termos.

print('='*30)
print('PROGRESSÃO ARITMÉTICA')
print('='*30)

a1 = int(input('Primeiro termo: '))
r = int(input('Razão: '))
i = 1
buffer = 0
termos = 0
user = ''
print('='*30)
print('PRIMEIROS 10 TERMOS DESTA PA')
print('='*30)

while i <= 10:
    print('Termo a{} é igual a {}'.format(i, a1))
    a1 = a1 + r
    i += 1
print('='*30)
user = input('Deseja visualizar mais termos? [S/N]: ').upper()
while user == 'S':
    termos = int(input('Quantos termos? '))
    while buffer < termos:
        print('Termo a{} é igual a {}'.format(i, a1))
        a1 = a1 + r
        i += 1
        buffer += 1
    print('='*30)
    user = input('Deseja visualizar mais termos? [S/N]: ').upper()
    buffer = 0
print('\nProgressão finalizada com {} termos mostrados'.format(i-1))


