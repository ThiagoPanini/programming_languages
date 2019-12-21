"""2.14 Faça um Programa que leia um número e exiba o dia correspondente da semana. (1-Domingo, 2- Segunda, etc.),
se digitar outro valor deve aparecer valor inválido"""

n = int(input('Digite um número: '
              '\n[ 1 ] DOMINGO'
              '\n[ 2 ] SEGUNDA-FEIRA'
              '\n[ 3 ] TERÇA-FEIRA'
              '\n[ 4 ] QUARTA-FEIRA'
              '\n[ 5 ] QUINTA-FEIRA'
              '\n[ 6 ] SEXTA-FEIRA'
              '\n[ 7 ] SÁBADO \n'))

if n == 1:
    print('DOMINGO selecionado!')
elif n == 2:
    print('SEGUNDA-FEIRA selecionada!')
elif n == 3:
    print('TERÇA-FEIRA selecionada!')
elif n == 4:
    print('QUARTA-FEIRA selecionada!')
elif n == 5:
    print('QUINTA-FEIRA selecionada!')
elif n == 6:
    print('SEXTA-FEIRA selecionada!')
elif n == 7:
    print('SÁBADO selecionado!')
else:
    print('Valor inválido. Encerrando programa.')