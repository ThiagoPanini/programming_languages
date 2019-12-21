"""2.16   O algoritmo deve mostrar na tela as notas, a média, o conceito correspondente
e a mensagem “APROVADO” se o conceito for A, B ou C ou “REPROVADO” se o conceito for D ou E. """

n1 = float(input('Primeira nota: '))
n2 = float(input('Segunda nota: '))
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

if conceito in 'ABC':
    print('Aluno APROVADO!')
if conceito in 'DF':
    print('Aluno REPROVADO!')