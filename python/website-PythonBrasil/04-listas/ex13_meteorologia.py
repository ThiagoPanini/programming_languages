"""4.13 Faça um programa que receba a temperatura média de cada mês do ano e  armazene-as em uma lista. Após isto,
calcule a média anual das temperaturas e mostre todas as temperaturas acima da média anual, e em que mês elas ocorreram
 (mostrar o mês por extenso: 1 – Janeiro, 2 – Fevereiro, . . . )."""

"""Faça um programa que receba a temperatura média de cada mês do ano e
armazene-as em uma lista. Após isto, calcule a média anual das temperaturas e
mostre todas as temperaturas acima da média anual, e em que mês elas ocorreram
(mostrar o mês por extenso: 1 – Janeiro, 2 – Fevereiro, . . . )."""

bdgeral = {}


class CorPy:
   azul = '\033[1;34m'
   end = '\033[m'


def cabecalho(texto):
   print('-'*35)
   print('{:^44}'.format(CorPy.azul + texto + CorPy.end))
   print('-'*35)


def preencheDados(ano):
    dicionario = {}
    listaTemp = []
    listaRH = []
    meses = 5
    temp = rh = 0
    cabecalho('Dados de Temperatura de ' + str(ano))
    for i in range(meses):
       item = float(input(f'Insira o valor de Temperatura do mês {i+1}: '))
       listaTemp.append(item)
       temp += item
    mediaTemp = temp/meses
    dicionario['Temperatura'] = listaTemp
    cabecalho('Dados de Umidade Relativa de ' + str(ano))
    for c in range(meses):
       umidade = float(input(f'Insira o valor de Umidade Relativa do mês {c+1}: '))
       listaRH.append(umidade)
       rh += umidade
    mediaRH = rh/meses
    dicionario['Umidade Relativa'] = listaRH
    return dicionario


while True:
   cabecalho('Preenchimento de Dados')
   ano = int(input('Insira o ano: '))
   bdaux = preencheDados(ano)
   bdgeral[ano] = bdaux
   r = input('Deseja continuar? [S/N]: ').strip().upper()
   if r in 'N':
       break

print(bdgeral)



