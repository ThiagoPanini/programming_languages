"""2.15 Faça um programa que lê as duas notas parciais obtidas por um aluno numa disciplina ao longo de um semestre,
e calcule a sua média. A atribuição de conceitos obedece à tabela abaixo:
      Media           Conceito
      Entre 9.0 e 10.0   A
      Entre 7.5 e 9.0   B
      Entre 6.0 e 7.5   C
      Entre 4.0 e 6.0   D
      Entre 4.0 e zero   E"""

n1 = float(input('Nota da Prova 1: '))
n2 = float(input('Nota da Prova 2: '))
media = (n1+n2)/2

if 9 < media <= 10:
    conceito = 'A'
elif 7.5 < media <= 9:
    conceito = 'B'
elif 6.0 < media <= 7.5:
    conceito = 'C'
elif 4.0 < media <= 6.0:
    conceito = 'D'
elif media <= 4.0:
    conceito = 'F'

print(f'Média: {media:.2f}. Conceito: {conceito}')