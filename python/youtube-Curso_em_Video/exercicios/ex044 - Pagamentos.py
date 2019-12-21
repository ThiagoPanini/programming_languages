#Desafio 44
#Elabore um programa que calcule o valor a ser pago por um produto, considerando seu preço normal e a condição de pagamento
# - À vista "dinheiro/cheque": 10% de desconto;
# - À vista no "cartão": 5% de desconto;
# - Em até "2x no cartão": preço normal;
# - "3x ou mais" no cartão: 20% de juros.

preco = float(input('Insira o preço do produto: R$'))
cond = int(input('Qual a condição de pagamento?'
                 '\n[ 1 ] para pagamento à vista em dinheiro ou cheque.'
                 '\n[ 2 ] para pagamento à vista no cartão.'
                 '\n[ 3 ] para pagamento em até 2x no cartão.'
                 '\n[ 4 ] para pagamento em 3x ou mais no cartão.'
                 '\nDigite: '))
if cond != 1 and cond != 2 and cond != 3 and cond != 4:
    print('Condição INVÁLIDA. Encerrando contato...')
else:
    if cond == 1:
        print('Condição [ 1 ] selecionada: Pagamento À VISTA em DINHEIRO/CHEQUE.')
        print('Valor final do produto (10% de desconto): R${:.2f}'.format(preco-preco*10/100))
    elif cond == 2:
        print('Condição [ 2 ] selecionada: Pagamento À VISTA no CARTÃO.')
        print('Valor final do produto (5% de desconto): R${:.2f}'.format(preco-preco*5/100))
    elif cond == 3:
        print('Condição [ 3 ] selecionada: Pagamento em até 2X no CARTÃO.')
        print('Valor final do produto (sem alteração): R${:.2f}'.format(preco))
        print('Valor da parcela em 2x: R${:.2f}'.format(preco/2))
    elif cond == 4:
        print('Condição [ 4 ] selecionada: Pagamento em 3X OU MAIS no CARTÃO.')
        print('Valor final do produto (20% de juros): R${:.2f}'.format(preco+preco*20/100))
        parcelas = int(input('Insira a quantidade de parcelas: '))
        print('Valor da parcela em 3x: R${:.2f}'.format((preco+preco*20/100)/parcelas))