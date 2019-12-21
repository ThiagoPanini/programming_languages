"""4.3 Faça um Programa que leia 4 notas, mostre as notas e a média na tela."""

lista = []
soma = 0
for item in range(4):
    x = float(input('Insira a {}ª nota: '.format(item+1)))
    lista.append(x)
    soma += x
print(lista)
media = soma/4
print(f'A média do aluno foi {media:.2f}')