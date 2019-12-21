#Deasfio 32
#Faça um programa que leia um ano qualquer e mostre se ele é BISSEXTO

from datetime import date #Biblioteca para datas: dias, meses e anos

ano = int(input('Digite um ano qualquer para verificar se ele é BISSEXTO. Digite 0 para o ano atual: '))
if ano == 0: #Se o usuário digitar 0
    ano = date.today().year #Comando para jogar o ano atual da máquina na variável 'ano'
if ano%4 == 0 and ano % 100 != 0 or ano%400 == 0: #!= operador diferente
    print('O ano {} é BISSEXTO'.format(ano))
else:
    print('O ano {} NÃO É BISSEXTO'.format(ano))