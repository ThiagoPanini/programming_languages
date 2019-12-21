"""Exercício 73
Crie uma tupla preenchida com os 20 primeiros colocados da Tabela do Campeonato Brasileiro,
na ordem de colocação. Depois mostre:
[ A ] Apenas os 5 primeiros colocados.
[ B ] Os últimos 4 colocados da tabela.
[ C ] Uma lista com os times em ordem alfabética.
[ D ] Em que posição da tabela está o time da Chapecoense."""

classificacao = ('São Paulo', 'Internacional', 'Flamengo', 'Grêmio', 'Palmeiras', 'Atlético MG', 'Cruzeiro',
                 'Corinthians', 'Fluminense', 'América MG', 'Bahia', 'Santos', 'Botafogo', 'Vasco', 'Chapecoense',
                 'Sport', 'Vitória', 'Atlético PR', 'Ceará', 'Paraná')

def cabecalho(option, arg):
    print('\n')
    print('- ' * 15)
    print('[ {:^20}'.format(option + ' ] - ' + arg))
    print('- ' * 15)

cabecalho('A', '5 primeiros colocados')
for i in range(5):
    print('{} - {}'.format(i+1, classificacao[i]))

cabecalho('B', 'Últimos 4 colocados')
for i in range(-4, 0):
    print('{} - {}'.format((classificacao.index(classificacao[i])+1), classificacao[i]))

cabecalho('C', 'Ordem alfabética')
print(sorted(classificacao))

cabecalho('D', 'Posição da Chape')
print('A Chapecoense está em {}ª no Campeonato Brasileiro'.format(classificacao.index('Chapecoense')+1))