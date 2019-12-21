"""3.31 O Sr. Manoel Joaquim expandiu seus negócios para além dos negócios de 1,99 e agora possui uma loja de conveniências.
Faça um programa que implemente uma caixa registradora rudimentar. O programa deverá receber um número desconhecido de
valores referentes aos preços das mercadorias. Um valor zero deve ser informado pelo operador para indicar o final da compra.
O programa deve então mostrar o total da compra e perguntar o valor em dinheiro que o cliente forneceu,
para então calcular e mostrar o valor do troco. Após esta operação, o programa deverá voltar ao ponto inicial,
para registrar a próxima compra. A saída deve ser conforme o exemplo abaixo:
Lojas Tabajara
Produto 1: R$ 2.20
Produto 2: R$ 5.80
Produto 3: R$ 0
Total: R$ 9.00
Dinheiro: R$ 20.00
Troco: R$ 11.00"""

soma = i = juros = 0
r = 'S'
lista = []

print('-'*15)
print('LOJAS TABAJARA')
print('-'*15)

while r in 'Ss':
    i += 1
    preco = float(input(f'Preço do Produto {i}: R$'))

    while preco < 0:
        preco = float(input('Insira um preço válido para o Produto {i}: R$'))

    soma += preco
    r = input('Há mais produtos na lista? [S/N] ').strip().upper()

    while r not in 'SN':
        r = input('Entrada inválida. Digite [S ou N]: ').strip().upper()

    tupla = (f'Produto {i}', f'{preco:.2f}')
    lista.append(tupla)

print('\nLista de compras:')
for item in range(len(lista)):
    print(f'{lista[item][0]} - R${lista[item][1]}')
print('-'*20)
print(f'Total: R${soma:.2f}')

pagamento = float(input('\nValor a ser pago: R$'))
if pagamento < soma:
    faltante = soma - pagamento
    opcao = int(input(f'Faltam R${faltante:.2f}, por favor selecione uma opção abaixo: \n'
          f'[ 1 ] Pagar a quantia de R${faltante:.2f} agora\n'
          f'[ 2 ] Informar data de pagamento (2% ao dia)\n'))
    if opcao == 1:
        pagamento = float(input(f'Insira a quantia R${faltante:.2f} faltante: R$'))
        while pagamento < faltante:
            faltante -= pagamento
            pagamento = float(input(f'Insira a quantia R${faltante:.2f} faltante: R$'))
        print(f'Pagamento de R${pagamento:.2f} confirmado.')
        if pagamento == faltante:
            print('Não deverá receber troco.')
            print('Fim.')
        else:
            troco = pagamento - faltante
            print(f'Deverá receber troco de R${troco:.2f}')
            print('Fim.')
    elif opcao == 2:
        data = int(input('Daqui quantos dias irá realizar o pagamento? '))
        for i in range(data):
            juros += (2/100)*faltante
        print(f'Valor a ser pago na data: R${faltante+juros:.2f}')
elif pagamento > soma:
    troco = pagamento - soma
    print(f'Troco: R${troco:.2f}')
    print('Fim.')
