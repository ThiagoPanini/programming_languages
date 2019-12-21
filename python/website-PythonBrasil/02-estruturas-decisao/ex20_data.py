"""2.20 Faça um Programa que peça uma data no formato dd/mm/aaaa e determine se a mesma é uma data válida. """
from datetime import date

dia = int(input('Digita o dia: '))
mes = int(input('Digite o mês: '))
ano = int(input('Digite o ano: '))

# Verificando se o ano é bissexto:
if ano%4 == 0 and ano % 100 != 0 or ano%400 == 0: #!= operador diferente
    bissexto = True
else:
    bissexto = False

# Validando data:
if dia > 31 or dia < 0 or mes < 1 or mes > 12 or ano > date.today().year:
    print('Data inválida! Encerrando programa.')
else:
    if dia > 30 and mes not in (1, 3, 5, 7, 8, 10, 12):
        print('Neste mês não há dia 31. Encerrando programa.')
    elif dia > 28 and mes == 2:
        if not bissexto:
            print(f'O ano de {ano} não é bissexto, portanto Fevereiro tem apenas 28 dias. Encerrando programa.')
    else:
        print(f'Data: {dia}/{mes}/{ano}')
