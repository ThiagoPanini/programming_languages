"""Exercício 89
Crie um programa que leia nome e duas notas de vários alunos e guarde tudo em uma lista composta.
No final, mostre um boletim contendo a média de cada um e permita que o usuário possa mostrar as notas
de cada aluno individualmente
"""

print('-'*25)
print(f'{"INSERINDO NOTAS":^25}')
print('-'*25)

classe = []
aluno = []
notas = []
cont = 0

while True:
    nome = input('Nome do aluno: ').capitalize()
    aluno.append(nome)
    n1 = float(input('Nota N1: '))
    notas.append(n1)
    n2 = float(input('Nota N2: '))
    notas.append(n2)
    media = (n1 + n2) / 2
    aluno.append(notas[:])
    aluno.append(media)
    notas.clear()
    classe.append(aluno[:])
    aluno.clear()
    r = input('Deseja continuar? [S/N] ').strip().upper()
    while r not in 'SN':
        r = input('Entrada inválida. Deseja continuar? [S/N] ').strip().upper()
    if r in 'N':
        break
    cont += 1

print(classe)

print('-'*25)
print(f'{"No.":<4}{"NOME":<10}{"MÉDIA":>8}')
print('-'*25)

for i, a in enumerate(classe):
    print(f'{i:<4}{a[0]:<10}{a[2]:>8.2f}')

while True:
    pesquisa = int(input('\nDigite o código do Aluno para pesquisar suas notas ou 999 para sair: '))
    if pesquisa == 999:
        break
    while pesquisa > len(classe) - 1:
        pesquisa = int(input('Aluno inexistente. Digite novamente: '))
        if pesquisa == 999:
            break
    print(f'Notas do aluno {classe[pesquisa][0]}: {classe[pesquisa][1]}')

print('\nENCERRANDO PROGRAMA')


