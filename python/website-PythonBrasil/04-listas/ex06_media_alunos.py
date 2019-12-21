"""4.6 Faça um Programa que peça as quatro notas de 10 alunos, calcule e armazene num vetor a média de cada aluno,
imprima o número de alunos com média maior ou igual a 7.0."""


def cabecalho(texto):
    print('-'*35)
    print('{:^44}'.format(CorPy.azul + texto.upper() + CorPy.end))
    print('-'*35)


def cadastroaluno():
    itens = []
    registro = int(input('Insira o RA: '))
    provas = int(input('Quantidade de provas: '))
    soma = 0
    for c in range(provas):
        n = float(input('Nota da prova {}: '.format(c+1)))
        soma += n
        itens.append(n)
    media = soma/provas
    return registro, itens, soma, media


class CorPy:
    azul = '\033[1;34m'
    negrito = '\033[1m'
    end = '\033[m'


cabecalho('Correção alunos')
for alunos in range(10):
    cabecalho(f'cadastro aluno {alunos+1}')
    ra, notas, soma, media = cadastroaluno()
    cabecalho(f'Dados do aluno {alunos+1}')
    print('RA: {}\nNotas: {}\nMédia: {}'.format(ra, notas, media))
cabecalho('fim')

