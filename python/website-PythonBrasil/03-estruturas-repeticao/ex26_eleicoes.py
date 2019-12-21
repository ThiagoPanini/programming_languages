"""3.26 Numa eleição existem três candidatos. Faça um programa que peça o número total de eleitores.
Peça para cada eleitor votar e ao final mostrar o número de votos de cada candidato."""

print('-'*27)
print('ELEIÇÕES 2018 - \033[1;31mLULA LIVRE\033[m')
print('-'*27)
lula = bolsonaro = marina = 0

eleitores = int(input('Número total de eleitores: '))

for c in range(1, eleitores+1):
    voto = int(input(f'Eleitor {c} digite seu voto: \n'
                     '[ 1 ] LULA\n'
                     '[ 2 ] BOLSONARO\n'
                     '[ 3 ] MARINA\n'))
    while voto not in [1, 2, 3]:
        voto = int(input('Voto inválido. Digite novamente: '))
    if voto == 1:
        lula += 1
    elif voto == 2:
        bolsonaro += 1
    elif voto == 3:
        marina += 1

print(f'LULA: {lula} votos.')
print(f'BOLSONARO: {bolsonaro} votos.')
print(f'MARINA: {marina} votos.')
