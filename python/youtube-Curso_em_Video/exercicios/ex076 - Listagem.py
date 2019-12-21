"""Crie um programa que tenha uma tupla única com nomes de produtos e seus
respectivos preços, na sequência.
No final, mostre uma listagem de preços, organizando os dados de forma tabular.
"""

listagem = ('Tagima', 500, 'OCD', 200, 'Peavey Bandit', 350)

print('-'*32)
print('{:^32}'.format('LISTAGEM DE PREÇOS'))
print('-'*32)
cont = 0

print('{:21}'.format('Produto'), end='Preço\n')
"""for i, j in zip(range(0, len(listagem), 2), range(1, len(listagem), 2)):
    print('{:.<20} R${:>8.2f}'.format(listagem[i], listagem[j]))"""

# Resolução Guanabara
for pos in range(0, len(listagem)):
    if pos % 2 == 0:
        print(f'{listagem[pos]:.<20}', end='')
    else:
        print(f'R${listagem[pos]:>7.2f}')

print('-'*32)





