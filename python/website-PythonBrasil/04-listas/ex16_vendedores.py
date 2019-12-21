"""4.16 Utilize uma lista para resolver o problema a seguir. Uma empresa paga seus vendedores com base em comissões.
O vendedor recebe $200 por semana mais 9 por cento de suas vendas brutas daquela semana. Por exemplo, um vendedor que
teve vendas brutas de $3000 em uma semana recebe $200 mais 9 por cento de $3000, ou seja, um total de $470.
Escreva um programa (usando um array de contadores) que determine quantos vendedores receberam salários nos seguintes
intervalos de valores:
a. $200 - $299   b. $300 - $399   c. $400 - $499   d. $500 - $599   e. $600 - $699   f. $700 - $799   g. $800 - $899
h. $900 - $999   i. $1000 em diante
Desafio: Crie ma fórmula para chegar na posição da lista a partir do salário, sem fazer vários ''ifs'' aninhados."""


class CorPy:
    azul = '\033[1;34m'
    end = '\033[m'


def cabecalho(texto):
    print('-'*35)
    print('{:^44}'.format(CorPy.azul + texto.strip().upper() + CorPy.end))
    print('-'*35)


def cadastroVendedor(n, database):
    for x in range(n):
        cabecalho(f'Cadastro Vendedor {x+1} ')
        nome = input('Nome: ').strip().title()
        sobrenome = input('Sobrenome: ').strip().title()
        idade = int(input('Idade: '))
        filial = verificaFilial()
        dbaux = {'Nome': nome, 'Sobrenome': sobrenome, 'Idade': idade, 'Filial': filial}
        database[101+x] = dbaux


def verificaFilial():
    filiais = ['Matriz', 'Centro', 'Shopping']
    loja = int(input('Filial: \n[ 1 ] Matriz\n[ 2 ] Centro\n[ 3 ] Shopping\n'))
    while loja not in range(1, len(filiais) + 1):
        loja = int(input('Filial inexistente. Digite: \n[ 1 ] Matriz\n[ 2 ] Centro\n[ 3 ] Shopping\n'))
    loja = filiais[loja-1]
    return loja


def vendasSemana(n, database):
    cabecalho('Vendas Semanais')
    for x in range(n):
        vendas = []
        soma = comissao = 0
        fixo = 1000
        cabecalho('Vendedor {}: {}'.format(101+x, database[101+x]['Nome'] + ' ' + database[101+x]['Sobrenome']))
        for semana in range(4):
            valor = float(input(f'Vendas da semana {semana+1}: R$'))
            vendas.append(valor)
            soma += valor
            comissao += (9/100)*valor
        database[101+x]['Vendas'] = vendas
        print('{}: R${:.2f}'.format(CorPy.azul + 'Total de Vendas' + CorPy.end, soma))
        print('{}: R${:.2f}'.format(CorPy.azul + 'Comissão' + CorPy.end, comissao))
        print('{}: R${:.2f}'.format(CorPy.azul + 'Salário' + CorPy.end, fixo + comissao))


bd = {}
cabecalho('Lojas Python')
n = int(input('Quantos vendedores serão cadastrados? '))
cadastroVendedor(n, bd)
vendasSemana(n, bd)
print(bd)