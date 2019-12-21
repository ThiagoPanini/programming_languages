"""3.25 Faça um programa que peça para n pessoas a sua idade, ao final o programa devera
verificar se a média de idade da turma varia entre 0 e 25 e 60 e maior que 60; e então,
dizer se a turma é jovem, adulta ou idosa, conforme a média calculada. """

r = 'Ss'
soma = i = 0

while r in 'Ss':
    i += 1
    idade = int(input(f'Idade da {i}ª pessoa: '))
    while idade < 0:
        idade = int(input(f'Digite uma idade válida para a {i}ª pessoa: '))
    soma += idade
    r = input('Deseja continuar? [S/N] ').strip().upper()[0]
    while r not in 'SN':
        r = input('Entrada inválida. Digite [S/N]: ').strip().upper()[0]
    if r in 'Nn' and i == 1:
        r = input('Apenas uma idade digitada. Insira pelo menos mais uma [S/N]:  ').strip().upper()
        while r not in 'SN':
            r = input('Entrada inválida. Digite [S/N]: ').strip().upper()[0]
media = soma / i
print(f'A média de idade das {i} pessoas é de {media:.0f} anos.')

if media <= 25:
    print('Classificação da turma: JOVEM.')
elif media <= 60:
    print('Classificação da turma: ADULTA.')
else:
    print('Classificação da turma: IDOSA.')


