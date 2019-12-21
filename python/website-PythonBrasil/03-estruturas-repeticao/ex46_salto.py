"""Em uma competição de salto em distância cada atleta tem direito a cinco saltos. No final da série de saltos de cada
atleta, o melhor e o pior resultados são eliminados. O seu resultado fica sendo a média dos três valores restantes.
Você deve fazer um programa que receba o nome e as cinco distâncias alcançadas pelo atleta em seus saltos e depois
informe a média dos saltos conforme a descrição acima informada (retirar o melhor  e o pior salto e depois calcular a
média). Faça uso de uma lista para armazenar os saltos. Os saltos são informados na ordem da execução, portanto não são
ordenados. O programa deve ser encerrado quando não for informado o nome do atleta.  A saída do programa deve ser
conforme o exemplo abaixo:"""

bdgeral = {}


def main():
    cabecalho('OLIMPÍADAS PYTHONICAS 2018')
    competicao = int(input('Digite o código da competição: \n'
                           '[ 1 ] Salto em distância\n'
                           '[ 2 ] Salto com vara\n'
                           '[ 3 ] Revezamento 4 por 4\n'
                           '[ 4 ] 50 metros livre\n'))
    while competicao not in range(1, 5):
        competicao = int(input('Entrada inválida. Digite novamente: '))
    if competicao == 1:
        saltocomvara()
    """elif competicao == 2:
        saltoemdistancia()
    elif competicao == 3:
        revezamento()
    elif competicao == 4:
        metroslivre()"""  # Não implementado. Treinamento para o futuro.


def saltocomvara():
    global bdgeral
    maior = menor = 0
    cabecalho('COMPETIÇÃO: SALTO EM DISTÂNCIA')
    while True:
        soma = 0
        cabecalho('CADASTRO DE COMPETIDOR')
        nome = input('Nome do atleta: ').strip().title()
        cod = int(input('Código do atleta: '))
        while cod in bdgeral:
            cod = int(input('Código pertecente ao atleta {}. Digite um novo código: '.format(bdgeral[cod]['Nome'])))
        n = int(input('Quantidade de saltos: '))
        bdgeral[cod] = {'Nome': nome, 'Saltos': [], 'Resultado': 0}
        for i in range(n):
            salto = float(input(f'Salto número {i+1} (em metros): '))
            if i == 0:
                maior = menor = salto
            else:
                if salto > maior:
                    maior = salto
                if salto < menor:
                    menor = salto
            bdgeral[cod]['Saltos'].append(salto)
            soma += salto
        resultado = (soma-maior-menor)/(n-2)
        print(f'Saltos descartados:\n'
              f'Maior: {maior}\n'
              f'Menor: {menor}\n'
              f'{CorPy.azul}Média:{CorPy.end} {resultado:.1f}')
        bdgeral[cod]['Resultado'] = round(resultado, 1)
        continua = input('\nCadastrar mais atletas? [S/N]: ').strip().upper()[0]
        if continua in 'N':
            break
    imprimebanco('Código', 'Nome', 'Salto', 'Resultado')
    print('Fim do programa')


def cabecalho(texto):
    print('-'*35)
    print('{:^42}'.format(CorPy.azul + texto + CorPy.end))
    print('-'*35)


def imprimebanco(title1, title2, title3 = '', title4 = ''):
    global bdgeral
    banco = input('Deseja realizar buscas no Banco de Dados? [S/N]: ').strip().upper()[0]
    while banco in 'S':
        cabecalho('BANCO DE DADOS')
        print('{}' '{:>15}'.format(CorPy.azul + title1, title2 + CorPy.end))
        for item in bdgeral.keys():
            print('{:<11}' '{}'.format(item, bdgeral[item]['Nome']))
        opcao = int(input('\nInsira o código do atleta para visualizar os dados: '))
        while opcao not in bdgeral.keys():
            opcao = int(input('Código inexistente. Digite novamente: '))
        cabecalho('DADOS DO ATLETA ' + str(opcao) + ' - ' + bdgeral[opcao]['Nome'])
        print('{}' '{:^19}'.format(CorPy.azul + title3, 'Distância' + CorPy.end))
        for valor in range(len(bdgeral[opcao]['Saltos'])):
            print('{:^5}' '{:>5} metros'.format(valor+1, bdgeral[opcao]['Saltos'][valor]))
        print(CorPy.verde + 'Resultado: ' + CorPy.end + '{} metros'.format(bdgeral[opcao]['Resultado']))
        banco = input('\nRealizar outra busca? [S/N]: ').strip().upper()[0]


class CorPy:
    azul = '\033[34m'
    negrito = '\033[1m'
    negativo = '\033[1;40m'
    vermelho = '\033[1;31m'
    verde = '\033[1;32m'
    end = '\033[m'


main()

"""Para o futuro: 
1 - Implementar ranking e ordenar os atletas de acordo com os Resultados.
2 - Ao mostrar o Banco de Dados, indicar em vermelho quais foram os saltos descartados."""