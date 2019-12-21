"""3.33 O Departamento Estadual de Meteorologia lhe contratou para desenvolver um programa que leia as um conjunto
indeterminado de temperaturas, e informe ao final a menor e a maior temperaturas informadas, bem como a média das
temperaturas."""

r = 'S'
soma = i = maior = menor = 0

while r in 'Ss':
    i += 1
    temperatura = float(input(f'Digite a {i}ª temperatura em °C: '))
    if i == 1:  # Se for a primeira verificação...
        maior = menor = temperatura
    while temperatura < 0:
        temperatura = float(input('Temperatura inválida. Digite novamente: '))

    if temperatura > maior:
        maior = temperatura
    if temperatura < menor:
        menor = temperatura

    soma += temperatura
    r = input('Deseja continuar? [S/N] ').strip().upper()[0]

    while r not in 'SN':
        r = input('Entrada inválida. Digite S ou N: ').strip().upper()[0]
media = soma / i

print(f'A maior temperatura foi de {maior}°C')
print(f'A menor temperatura foi de {menor}°C')
print(f'A média das {i} temperaturas foi de {media}°C')