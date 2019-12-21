#Desafio 2 - Data de nascimento e informações adicionais

nome = input('Qual o seu nome? ')
print('É um grande prazer te conhecer,',nome)
idade = input('Qual a sua idade? ')
peso = input('Obrigado, agora preciso saber seu peso ')
print('Muito obrigado pelas informações, registramos que o(a)', nome, 'possui', idade,'anos e tem', peso,'kilos')

print('Agora precisarei saber a data de seu nascimento...')
dia = input('Digite o dia ')
mes = input('Digite o mês ')
ano = input('Digite o ano ')

print('Identificamos que você nasceu em', dia,'de',mes,'de',ano, 'ou',dia,'/',mes,'/',ano)
