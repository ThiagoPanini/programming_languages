"""3.41 Faça um programa que receba o valor de uma dívida e mostre uma tabela com os seguintes dados: valor da dívida,
valor dos juros, quantidade de parcelas e valor da parcela. """

print('-'*27)
print('{:-^27}'.format(' TABELA DE JUROS '))
print('-'*27)

print('{:<15}' '{:^15}'.format('Quantidade de', '% de Juros'))
print('{:^14}'.format('parcelas'))


def imprimeTabela(maxparcela):
    i = 0
    for c in range(1, maxparcela+2, 3):
        if c >= 4:
            c -= 1
        print('{:^12}' '{:^15}'.format(c, i))
        if i >= 10:
            i += 5
        else:
            i += 10


def imprimeDivida(valor, maxparcelas):
    i = 0
    inicial = valor
    for c in range(1, maxparcelas+2, 3):
        if c >= 4:
            c -= 1
        valor = inicial + inicial*(i/100)
        print('R${:^12.2f}' 'R${:^12.2f}' '{:^9}' '{:5}' 'R${:^16.2f}'.format(valor, valor-inicial, c, '', (valor/c)))
        if i >= 10:
            i += 5
        else:
            i += 10


# Programa Principal
imprimeTabela(12)
divida = float(input('\nInsira o valor da dívida: R$'))
print('-'*60)
print('{:^15}' '{:<12}' '{:<16}' '{:<15}'.format('Vlr Dívida', 'Vlr Juros', 'Qtd Parcelas', 'Vlr Parcela'))
imprimeDivida(divida, 12)

