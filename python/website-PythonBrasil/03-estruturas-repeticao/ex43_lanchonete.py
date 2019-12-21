"""3.43   Faça um programa que leia o código dos itens pedidos e as quantidades desejadas. Calcule e mostre o valor a
ser pago por item (preço * quantidade) e o total geral do pedido. Considere que o cliente deve informar quando o pedido
deve ser encerrado. """

menu = [('Cachorro Quente', 100, 1.20),
        ('Bauru Simples', 101, 1.30),
        ('Bauru com Ovo', 102, 1.50),
        ('Hambúrguer', 103, 1.20),
        ('Cheeseburguer', 104, 1.30),
        ('Refrigerante', 105, 1.00)]
pedido = []
r = ''
conta = itens = soma = i = 0


def cabecalho(texto):
    print('-'*40)
    print('{:-^40}'.format(texto))
    print('-'*40)


def imprimeCardapio(lista):
    i = 0
    print('\033[1;34m{:^16}' '{:^10}' '{:^14}'.format('Item', 'Código', 'Preço\033[m'))
    for c in range(len(lista)):
        print('{:<19}' '{:<9}' 'R$ {:^5.2f}'.format(lista[i][0], lista[i][1], lista[i][2]))
        i += 1


cabecalho(' MENU LANCHONETE PYTHON ')
imprimeCardapio(menu)
cabecalho(' FAÇA SEU PEDIDO ')
while True:
    cod = int(input('O que deseja? Insira o código aqui: '))
    qtd = int(input('Quantidade: '))
    for (a, b, c) in menu:
        if cod == b:
            item = (a, qtd, qtd*c)
            pedido.append(item)
            break
    r = input('Mais alguma coisa? [S/N]: ').strip().upper()[0]
    while r not in 'SN':
        r = input('Digite [ S ] para continuar e [ N ] para encerrar pedido: ').strip().upper()[0]
    if r in 'Nn':
        break

cabecalho(' IMPRIMINDO PEDIDO ')
print('\033[1;34m{:^14}' '{:^9}' '{:^11}'.format('Item', 'Qtd', 'Preço\033[m'))
for (x, y, z) in pedido:
    print('{:<18}' '{:<5}' 'R$ {:<6.2f}'.format(x, y, z))
    soma += z
print('{:>30}'.format('-'*7))
print('\033[1;34mTotal: ' '{:16}' '\033[mR$ {:.2f}'.format('', soma))

cabecalho(' PAGAMENTO ')
pag = int(input('Insira a forma de pagamento: \n'
                '[ 1 ] Dinheiro\n'
                '[ 2 ] Cartão\n'))
if pag == 1:
    valor = float(input('Valor a ser pago: R$ '))
    while valor < soma:
        soma -= valor
        print(f'Valor insuficiente. Faltam R$ {soma:.2f}')
        valor = float(input('Insira o valor faltante: R$ '))
    troco = valor - soma
    print(f'Valor pago: R$ {valor:.2f}\n'
          f'Troco: R$ {troco:.2f}\n'
          f'\033[1;34mVolte sempre!\033[m')
if pag == 2:
    cartao = int(input('Insira a modalidade: \n'
                       '[ 1 ] Débito\n'
                       '[ 2 ] Crédito\n'))
    if cartao == 1:
        print(f'Débito de R$ {soma:.2f} descontado automaticamente. \n\033[1;34mVolte sempre!\033[m')
    if cartao == 2:
        print(f'Crédito selecionado. Valor de R$ {soma:.2f} descontado no cartão. \n\033[1;34mVolte sempre!\033[m')







"""Eu usaria uma lista de dicionários. Ex: [{"Cachorro quente ":{"codigo:101, "preco": 1.20}]"""


