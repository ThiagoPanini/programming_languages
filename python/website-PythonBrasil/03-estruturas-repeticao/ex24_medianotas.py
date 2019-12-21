"""3.24 Faça um programa que calcule o mostre a média aritmética de N notas.  """

r = 'Ss'
i = soma = 0

print('-'*25)
print('Média Aritmética de Notas')
print('-'*25)

while r in 'Ss':
    i += 1
    n = float(input(f'Insira a {i}ª nota: '))
    soma += n
    r = input('Deseja continuar? [S/N] ').strip().upper()[0]
    while r in 'Nn' and i == 1:
        r = input('Não é possível calcular média com apenas uma nota. Por favor, insira pelo menos mais 1 nota: [S/N] ')
media = soma/i

print(f'A média das {i} notas é {media:.2f}')