"""3.40 Foi feita uma estatística em cinco cidades brasileiras para coletar dados sobre acidentes de trânsito.
Foram obtidos os seguintes dados:
a. Código da cidade;
b. Número de veículos de passeio (em 1999);
c. Número de acidentes de trânsito com vítimas (em 1999).
Deseja-se saber:
d. Qual o maior e menor índice de acidentes de transito e a que cidade pertence;
e. Qual a média de veículos nas cinco cidades juntas;
f. Qual a média de acidentes de trânsito nas cidades com menos de 2.000 veículos de passeio."""

indexcodigos, indexcidades, indexveiculos, indexacidentes = 0, 1, 2, 3

maior = menor = i = 0

codigos = []
cidades = []
veiculos = []
acidentes = []

j = int(input('Quantas cidades serão analisadas? '))


def entrada(string, n, lista):
    if lista == cidades:
        saida = input(f'Digite o {string} da {n}ª cidade: ').strip().title()
    else:
        saida = int(input(f'Digite o {string} da {n}ª cidade: '))
    if lista == codigos or lista == cidades:
        while saida in lista:
            saida = input(f'Uma chave primária não pode se repetir\nDigite novamente o {string} da {n}ª cidade: ')
    lista.append(saida)


def verifica(lista, index):
    i = indicemaior = indicemenor = soma = maiortaxa = menortaxa = 0
    while i < j:
        item = lista[i][index]
        item2 = lista[i][index-1]
        if i == 0 and index == 3:
            taxa = item/item2
            maiortaxa = menortaxa = taxa*100
            indicemaior = indicemenor = i
        if i != 0 and index == 3:
            taxa = item/item2
            if taxa*100 > maiortaxa:
                maiortaxa = taxa*100
                indicemaior = i
            if taxa*100 < menortaxa:
                menortaxa = taxa*100
                indicemenor = i
        if index == 2:
            soma += item
        i += 1
    return maiortaxa, menortaxa, soma, indicemaior, indicemenor


for c in range(1, j+1):
    print('-'*35)
    entrada('código', c, codigos)
    entrada('nome', c, cidades)
    entrada('número de veículos', c, veiculos)
    entrada('acidentes', c, acidentes)

bd = list(zip(codigos, cidades, veiculos, acidentes))
print('-'*45)
print('{:~^45}'.format('BANCO DE DADOS'))
print('-'*45)
print('{:<12}' '{:<12}' '{:<12}' '{:<12}'.format('Código', 'Cidade', 'Veículos', 'Acidentes'))

for c in range(j):
    for h in range(j+1):
        print('{:<12}'.format(bd[c][h]), end=' ')
    print('\n')

geralAcidentes = verifica(bd, indexacidentes)
geralVeiculos = verifica(bd, indexveiculos)
maiortaxa, menortaxa, soma, indicemaior, indicemenor = 0, 1, 2, 3, 4
print(f'\nSomatório de veículos em todas as cidades: {geralVeiculos[soma]}.')
print(f'Média de veículos por cidade: {geralVeiculos[soma]/j:.2f}')
print(f'\nMaior taxa de acidentes: {geralAcidentes[maiortaxa]:.2f}% - Em {bd[geralAcidentes[indicemaior]][1]}.')
print(f'Menor taxa de acidentes: {geralAcidentes[menortaxa]:.2f}% - Em {bd[geralAcidentes[indicemenor]][1]}.')




