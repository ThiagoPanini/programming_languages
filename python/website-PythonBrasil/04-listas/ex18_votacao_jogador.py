"""4.18 Votação_Jogadores"""

import random


class CorPy:
    azul = '\033[1;34m'
    vermelho = '\033[1;31m'
    turquesa = '\033[1;36m'
    amarelo = '\033[1;33m'
    verde = '\033[1;32m'
    end = '\033[m'


def cabecalho(texto):
    """Imprime cabeçalho de título"""
    print(CorPy.azul)
    print('-'*35, end='')
    print(CorPy.end)
    print('{:^44}'.format(CorPy.amarelo + texto.upper() + CorPy.end), end='')
    print(CorPy.azul)
    print('-'*35)
    print(CorPy.end, end='')


def cabecalhoPartida():
    """Imprime cabeçalho da partida de futebol"""
    print(CorPy.azul)
    print('-'*35, end='')
    print(CorPy.end)
    print('{:^52}'.format(CorPy.amarelo + 'BRASIL' + CorPy.end + ' 2 x 0 '
                          + CorPy.vermelho + 'SUIÇA' + CorPy.end), end='')
    print(CorPy.azul)
    print('-'*35, end='')
    print(CorPy.end, end='')


def imprimeJogadores(dicionario):
    """Imprime lista de jogadores da seleção brasileira"""
    cabecalho('man of the match')
    for item in dicionario:
       print('{:5}''{:<18}''{:>3}'.format('', dicionario[item][0], item))


def votacao(dicionario):
    """Organiza e computa votos Manuais ou Automáticos (sorteio)"""
    cabecalho('votação')
    cont = 0
    modalidade = int(input('Insira uma opção: \n[ 1 ] Voto humano\n[ 2 ] Voto automático\n'))
    while modalidade not in range(1, 3):
        modalidade = int(input('Opção inválida. Digite: \n[ 1 ] Voto humano\n[ 2 ] Voto automático\n'))
    if modalidade == 1:
        while True:
            voto = int(input('Insira o número da camisa do jogador para votar [0 para sair]: '))
            if voto == 0:
                break
            while voto not in dicionario:
                voto = int(input('Voto inválido. Digite novamente: '))
            cont += 1
            dicionario[voto][1] += 1
    elif modalidade == 2:
        cont = int(input('Quantidade de participantes: '))
        for num in range(cont):
            dicionario[random.choice(list(dicionario.keys()))][1] += 1  # Sorteia votos (chaves) e adiciona na lista
        return cont


def resultado(dicionario, participantes):
    """Imprime os resultados da votação"""
    cabecalho('resultado')
    vencedor = chave = 0
    print('{}{:<5}{:<15}{:>3}{:>5}{}'.format(CorPy.amarelo, 'Nº', 'Nome', 'Votos', '%', CorPy.end))
    for item in dicionario:
        print('{:<5}{:<17}{:<4}{:>5.1f}'.format(item, dicionario[item][0], dicionario[item][1],
                                                dicionario[item][1]*100/participantes))
        if item == 1:
            chave = item
            vencedor = dicionario[chave][1]
        else:
            if dicionario[item][1] > vencedor:
                chave = item
                vencedor = dicionario[item][1]
    print(f'\nQuantidade de votos: {participantes}')
    print('O jogador {} foi eleito como melhor da partida com {} votos!'
          .format(CorPy.azul + dicionario[chave][0] + CorPy.end, vencedor))


titulares = {1: ['Alisson', 0], 22: ['Fagner', 0], 12: ['Marcelo', 0], 3: ['Miranda', 0], 14: ['Thiago Silva', 0],
             5: ['Casemiro', 0], 8: ['Renato Augusto', 0], 15: ['Paulinho', 0],
             10: ['Neymar', 0], 9: ['Gabriel Jesus', 0], 11: ['William', 0]}

cabecalhoPartida()
imprimeJogadores(titulares)
n = votacao(titulares)
resultado(titulares, n)

# print(titulares)

"""Cadastrar todos os jogos da primeira fase e gravar em arquivo para companhar e atualizar durante a Copa"""

