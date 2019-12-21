"""3.44 Em uma eleição presidencial existem quatro candidatos. Os votos são informados por meio de código"""


class CorPy:
    titulo = '\033[34m'
    negrito = '\033[1m'
    negativo = '\033[1;40m'
    vermelho = '\033[1;31m'
    verde = '\033[1;32m'
    end = '\033[m'


def cabecalho(titulo):
    print('-'*40)
    print('{:^48}'.format(CorPy.titulo + titulo + CorPy.end))
    print('-'*40)


def cadastro():
    dicionario = {}
    n = int(input('Quantidade de opções: '))
    for c in range(n+1):
        if c == n:
            dicionario[c+1] = ['Nulo', 0]
        else:
            dicionario[c+1] = [input(f'Opção {c+1}: ').title(), 0]
    print('\nCadastro realizado com ' + CorPy.verde + 'sucesso!' + CorPy.end)
    return dicionario


def imprime(dicionario, participantes, num):
    if num == 1:
        print('{:^29}' '{}'.format(CorPy.titulo + 'Opção', 'Item' + CorPy.end))
        for linha in range(len(dicionario)):
            print('{:^23}' '{}'.format(linha+1, dicionario[linha+1][0]))
    if num == 2:
        print('{:^15}' '{:^9}' '{:>15}' '{:>8}'.format(CorPy.titulo + 'Opção', 'Item', 'Votos', '%' + CorPy.end))
        for linha in range(len(dicionario)):
            print('{:^9}' '{:<22}' '{:<5}' '{:.2f}'.format(linha + 1, dicionario[linha + 1][0][:19],
                                                 dicionario[linha + 1][1], dicionario[linha + 1][1]*100/participantes))


def vota(dicionario):
    participantes = int(input('Quantidade de participantes: '))
    print('Para ver o resultado parcial, digite ' + CorPy.verde + '999' + CorPy.end + ' a qualquer momento.')
    for i in range(participantes):
        voto = int(input(f'Voto do {i+1}º participante: '))
        while voto == 999:
            print('-' * 40)
            imprime(dicionario, i, 2)
            print('-' * 40)
            voto = int(input(f'Voto do {i+1}º participante: '))
        while voto not in dicionario.keys() and voto != 999:
            voto = int(input(f'Voto inválido. Digite entre {1} e {len(dicionario)}: '))
            while voto == 999:
                print('-' * 40)
                imprime(dicionario, i, 2)
                print('-' * 40)
                voto = int(input(f'Voto do {i+1}º participante: '))
        dicionario[voto][1] += 1
    return participantes


while True:
    cabecalho(input('Tema da pesquisa: ').strip().upper())
    cabecalho('CADASTRO')
    bd = cadastro()
    cabecalho('APRESENTAÇÃO')
    imprime(bd, 0, 1)
    cabecalho('VOTAÇÃO')
    participantes = vota(bd)
    cabecalho('RESULTADO')
    imprime(bd, participantes, 2)
    r = input('\nDeseja fazer uma nova pesquisa? [S/N]: ').strip().upper()[0]
    if r in 'Nn':
        break
print('Obrigado pela participação!')
