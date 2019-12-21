#Desafio 39
#Faça um programa que leia o ano de nascimento de um jovem e informe, de acordo com sua idade, se ele ainda vai
#se alistar ao serviço militar, se é a hora de se alistar ou se já passou do tempo do alistamento.
#Seu programa também deverá mostrar o tempo que falta ou que passou do prazo.

from datetime import date

nasc = int(input('Qual seu ano de nascimento em 4 dígitos? '))
ano = date.today().year
idade = ano - nasc
if idade == 18:
    print('Você possui {} anos e neste ano de {} você deve se alistar. Boa sorte!'.format(idade, ano))
elif idade > 18:
    print('Você possui {} anos e deveria ter se alistado em {}. Está atrasado {} anos!'.format(idade, (ano-(idade-18)), (idade-18)))
else:
    print('Você possui {} anos e ainda faltam {} ano(s) para seu alistamento. Apresente-se em {}!'.format(idade, (18-idade), (18-idade+ano)))