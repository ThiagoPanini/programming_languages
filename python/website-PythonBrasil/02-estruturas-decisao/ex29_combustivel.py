"""2.29   Escreva um algoritmo que leia o número de litros vendidos, o tipo de combustível
(codificado da seguinte forma: A-álcool, G-gasolina), calcule e imprima o valor a ser pago pelo cliente sabendo-se que
 o preço do litro da gasolina é R$ 2,50 o preço do litro do álcool é R$ 1,90.
 Um posto está vendendo combustíveis com a seguinte tabela de descontos:
 a. Álcool:
 b. até 20 litros, desconto de 3% por litro
 c. acima de 20 litros, desconto de 5% por litro
 d. Gasolina:
 e. até 20 litros, desconto de 4% por litro
 f. acima de 20 litros, desconto de 6% por litro"""

print('~'*20)
print('POSTO PYTHON SHELL')
print('~'*20)

tipo = input('Com qual combustível deseja abastecer?\n'
             '[ A ] Álcool\n'
             '[ G ] Gasolina\n').strip().upper()[0]

modo = input('Como deseja quantificar o abastecimento?\n'
                   '[ L ] Em Litros\n'
                   '[ R ] Em Reais\n').strip().upper()[0]

if modo == 'R':
    reais = float(input('Quantos reais deseja abastecer? R$'))
    if tipo == 'G':
        litros = reais/2.50
        print(f'R${reais:.2f} resulta em {litros:.2f}L de Gasolina.')
    elif tipo == 'A':
        litros = reais/1.90
        print(f'R${reais:.2f} resulta em {litros:.2f}L de Álcool.')
elif modo == 'L':
    litros = float(input('Quantos litros deseja abastecer? '))
    if tipo == 'G':
        reais = litros*2.50
        print(f'{litros:.2f}L de Gasolina resulta em R${reais:.2f}')
    elif tipo == 'A':
        reais = litros*1.90
        print(f'{litros:.2f}L de Álcool resulta em R${reais:.2f}')

if tipo == 'A':
    if litros <= 20:
        print(f'DESCONTO ESPECIAL 3%: Até 20L de Álcool, o valor de R${reais:.2f} cairá para R${reais*0.97:.2f}')
    else:
        print(f'DESCONTO ESPECIAL 5%: 20L de Álcool ou mais, o valor de R${reais:.2f} cairá para R${reais*0.95:.2f}')
if tipo == 'G':
    if litros <= 20:
        print(f'DESCONTO ESPECIAL 4%: Até 20L de Gasolina, o valor de R${reais:.2f} cairá para R${reais*0.96:.2f}')
    else:
        print(f'DESCONTO ESPECIAL 6%: 20L de Gasolina ou mais, o valor de R${reais:.2f} cairá para R${reais*0.94:.2f}')


