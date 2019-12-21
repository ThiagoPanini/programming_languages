"""3.28 Faça um programa que calcule o valor total investido por um colecionador em sua coleção de CDs
e o valor médio gasto em cada um deles. O usuário deverá informar a quantidade de CDs e o valor para em cada um."""

cds = []
r = 'S'
soma = 0
while r in 'Ss':
    album = input('Nome do álbum: ').title()
    cds.append(album)
    r = input('Deseja continuar? [S/N] ')
for c in range(len(cds)):
    preco = float(input(f'Quanto custou o {cds[c]}? R$'))
    while preco < 0:
        preco = float(input(f'Valor inválido. Quanto custou o {cds[c]}? R$'))
    soma += preco
print(f'\nInvestimento total na coleção dos {len(cds)} álbuns: R${soma:.2f}')