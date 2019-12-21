"""3.45 Desenvolver um programa para verificar a nota do aluno em uma prova com 10 questões, o programa deve perguntar
ao aluno a resposta de cada questão e ao final comparar com o gabarito da prova e assim calcular o total de acertos e a
nota (atribuir 1 ponto por resposta certa). Após cada aluno utilizar o sistema deve ser feita uma pergunta se outro
aluno vai utilizar o sistema. Após todos os alunos terem respondido informar:
a. Maior  e Menor Acerto;
b. Total de Alunos que utilizaram o sistema;
c. A Média das Notas da Turma"""

bd = {}
banco = 0

class CorPy:
    titulo = '\033[34m'
    negrito = '\033[1m'
    negativo = '\033[1;40m'
    vermelho = '\033[1;31m'
    verde = '\033[1;32m'
    end = '\033[m'


def cabecalho(texto):
    print('-'*29)
    print('{:^37}'.format(CorPy.titulo + texto + CorPy.end))
    print('-'*29)


def preencheGabarito(n):
    gabarito = {}
    for c in range(n):
        r = input(f'Resposta da questão {c+1}: ').strip().upper()[0]
        gabarito[c+1] = r
    return gabarito


def imprimeGabarito(professor, n, gabaritoProf = {}, gabaritoAluno={}, id = 0):
    global banco
    if professor:
        cabecalho('Gabarito Oficial')
        print('{}' '{:>13}'.format(CorPy.titulo + 'Questão', 'Resposta' + CorPy.end))
        for item in range(n):
            print('{:^7}' '{:>6}'.format(item + 1, gabaritoProf[item + 1]))
    else:
        if banco == 0:
            cabecalho('Gabarito do Aluno')
            print('{}' '{:>9}' '{:^15}'.format(CorPy.titulo + 'Questão', 'Oficial', str(id) + CorPy.end))
            for item in range(n):
                print('{:^7}' '{:^11}' '{:>10}'.format(item + 1, gabaritoProf[item + 1],
                CorPy.verde + gabaritoAluno[item + 1] + CorPy.end if gabaritoAluno[item+1] == gabaritoProf[item+1] else
                CorPy.vermelho + gabaritoAluno[item+1] + CorPy.end))
        if banco == 1:
            cabecalho('Gabarito do Aluno')
            print('{}' '{:>9}' '{:^15}'.format(CorPy.titulo + 'Questão', 'Oficial', str(id) + CorPy.end))
            for item in range(n):
                print('{:^7}' '{:^11}' '{:>15}'.format(item + 1, gabaritoProf[item + 1],
                                                       CorPy.verde + bd.get(id)['Respostas'][item+1] + CorPy.end if
                                                       bd.get(id)['Respostas'][item + 1] == gabaritoProf[item + 1] else
                                                       CorPy.vermelho + bd.get(id)['Respostas'][item+1] + CorPy.end))


def verificaacertos(gabaritoProf, gabaritoAluno):
    cont = 0
    for questao in range(len(professor)):
        if professor[questao+1] == aluno[questao+1]:
            cont += 1
    return cont


# Programa Principal main()

# Cadastro e preenchimento referente ao PROFESSOR
cabecalho('Sistema de Correção de Provas')
cabecalho('Cadastro de Respostas')
n = int(input('Quantidade de questões: '))
professor = preencheGabarito(n)
imprimeGabarito(True, n, professor)
erro = input('\nDeseja fazer alguma correção? [S/N]: ').strip().upper()[0]
while True:
    if erro in 'N':
        break
    while erro in 'S':
        questao = int(input('Em qual questão? '))
        novo = input(f'Digite uma nova resposta para a questão {questao}: ').strip().upper()[0]
        professor[questao] = novo
        erro = input('Mais alguma correção? [S/N] ').strip().upper()
    imprimeGabarito(True, n, professor)

# Cadastro e preenchimento referente aos ALUNOS
while True:
    ra = int(input('\nDigite o RA do aluno: '))
    cabecalho('Respostas do Aluno ' + str(ra))
    aluno = preencheGabarito(n)
    imprimeGabarito(False, n, professor, aluno, ra)  # Teste
    acertos = verificaacertos(professor, aluno)
    print(f'Acertos do aluno {ra}: {acertos}')
    erro = input('\nDeseja fazer alguma correção? [S/N]: ').strip().upper()[0]
    while erro in 'S':
        questao = int(input('Em qual questão? '))
        novo = input(f'Digite uma nova resposta para a questão {questao}: ').strip().upper()[0]
        aluno[questao] = novo
        erro = input('Mais alguma correção? [S/N] ').strip().upper()
        if erro in 'N':
            imprimeGabarito(False, n, professor, aluno, ra)  # Teste
            acertos = verificaacertos(professor, aluno)
            print(f'Acertos do aluno {ra}: {acertos}')
    bd['Oficial'] = professor
    bd[ra] = {'Respostas': aluno, 'Acertos': acertos}
    corrigir = input('\nCorrigir mais provas? [S/N]: ').strip().upper()[0]
    if corrigir in 'N':
        break

# Busca no Banco de Dados
busca = input('\nDeseja buscar um resultado no Banco de Dados? [S/N]: ').strip().upper()[0]
if busca in 'S':
    cabecalho('Busca no Banco de Dados')
    banco = 1
while busca in 'S':
    chave = int(input('Digite o RA do aluno ou então [ 0 ] para ver o gabarito oficial: '))
    if chave == 0:
        imprimeGabarito(True, n, professor)
        while True:
            item = int(input(f'\nDigite o nº da questão (entre 1 e {n}) ou então [ 0 ] para sair: '))
            if item == 0:
                break
            while item not in range(1, n + 1):
                item = int(input(f'Questão inexistente. Digite entre 1 e {n}: '))
            print('Resposta oficial da questão {}: {}'.format(item, CorPy.verde + bd['Oficial'][item] + CorPy.end))
            searchq = input('\nBuscar outra questão? [S/N]: ').strip().upper()[0]
            if searchq in 'N':
                break
    else:
        while chave not in bd.keys():
            chave = int(input('Matrícula inexistente. Digite novamente o RA: '))
        imprimeGabarito(False, n, professor, aluno, chave)
        print('Acertos do aluno {}: {}.'.format(chave, bd[chave]['Acertos']))
    busca = input('\nRealizar outra busca? [S/N]: ').strip().upper()[0]
    if busca in 'N':
        break
print(professor)
print(aluno)
print(bd)

""" Problema encontrado: Na hora da busca no Banco de Dados, o programa mostra sempre a última sequencia de respostas do
último aluno, mesmo que a chave informada seja de qualquer outro aluno.
Solução: verificar na função "imprimeGabarito" sobre isso. Provavelmente lá está imprimindo erroneamente sempre os
últimos dados inputados.
Os acertos estão OK, ou seja, ele está trazendo baseado na chave inputada na busca. Apenas o gabarito está errado."""

"""RESOLVIDO"""

"""OBSERVAÇÃO: Acredito não ter dado certo pois na função "preencheGabarito", o gabarito está sendo inicializado como
gabarito = {}, ou seja, sempre que chamamos esta função, ela apaga os valores já inputados e insere novos valores. Por
isso, ao chamarmos a função imprimeGabarito, os dicionários referenciados possuem apenas um único valor, no caso, os 
últimos dados inseridos."""

