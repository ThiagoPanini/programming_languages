"""2.31   Para atender a todos os clientes, cada cliente poderá levar apenas um dos tipos de carne da promoção,
porém não há limites para a quantidade de carne por cliente. Se compra for feita no cartão Tabajara o cliente receberá
ainda um desconto de 5% sobre o total a compra. Escreva um programa que peça o tipo e a quantidade de carne comprada
pelo usuário e gere um cupom fiscal, contendo as informações da compra: tipo e quantidade de carne, preço total,
tipo de pagamento, valor do desconto e valor a pagar.
                    Até 5 Kg           Acima de 5 Kg
File Duplo      R$ 4,90 por Kg          R$ 5,80 por Kg
Alcatra         R$ 5,90 por Kg          R$ 6,80 por Kg
Picanha         R$ 6,90 por Kg          R$ 7,80 por Kg """

print('=-'*10)
print('Açougue PYTHON BEEF')
print('=-'*10)

"""print('Tabela de preços: \n'
      '{:>28}' '{:>20}\n' '{:<18}' '{:<18}' '{:<18}\n'
      '{:<18}' '{:<18}' '{:<18}\n'
      '{:<18}' '{:<18}' '{:<18}\n'
      .format('Até 5Kg', 'Acima de 5Kg', 'Filé Duplo', 'R$4,90 por Kg', 'R$5,80 por Kg',
              'Alcatra', 'R$5,90 por Kg', 'R$7,80 por Kg',
              'Picanha', 'R$6,90 por Kg', 'R$7,80 por Kg\n'), end='')

carnes = [('Filé Duplo', 'R$4.90 por Kg', 'R$5.80 por Kg'),
          ('Alcatra', 'R$5.90 por Kg', 'R$6.80 por Kg'),
          ('Picanha', 'R$6.90 por Kg', 'R$7.80 por Kg')]

i = 0
j = 0
for item in carnes[i]:
    print(f'{carnes[i][j]:<18}',end='')
    j += 1
    if j == 3:
        print('\n')
        j = 0
        i += 1
        for item in carnes[i]:
            print(f'{carnes[i][j]:<18}', end='')
            j += 1
            if j == 3:
                print('\n')
                j = 0
                i += 1
                for item in carnes[i]:
                    print(f'{carnes[i][j]:<18}', end='')
                    j += 1"""

carnes = [('Filé Duplo', 'R$4.90 por Kg', 'R$5.80 por Kg'),
          ('Alcatra', 'R$5.90 por Kg', 'R$6.80 por Kg'),
          ('Picanha', 'R$6.90 por Kg', 'R$7.80 por Kg')]

valores = [(4.90, 5.90, 6.90), (5.80, 6.80, 7.80)] # Lista formada por 2 Tuplas organizadas por: [0]<5Kg e [1]>5Kg

print('-'*50)
print('{:^50}'.format('FINALMENTE O JEITO CORRETO DE FAZER ...'))
print('-'*50)

i = j = 0
print('{:>28}' '{:>20}'.format('Até 5Kg', 'Acima de 5Kg'))
while i < 3:
    print(f'{carnes[i][j]:<18}', end='')
    j += 1
    if j == 3:
        print('\n')
        i += 1
        j = 0

tipo = int(input('Qual carne deseja levar? \n'
                 '[ 1 ] Filé Duplo\n'
                 '[ 2 ] Alcatra\n'
                 '[ 3 ] Picanha\n'))
kg = float(input('Quantos Kg de {}? '.format(carnes[tipo-1][0])))
index = 0 if kg <= 5 else 1
preco = valores[index][tipo-1]*kg
print(f'{kg}Kg de {carnes[tipo-1][0]} a {carnes[tipo-1][index+1]} é igual a: R${preco:.2f} ')
print('\nVolte sempre!')

