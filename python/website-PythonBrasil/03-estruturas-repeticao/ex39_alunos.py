"""3.39 Faça um programa que leia dez conjuntos de dois valores, o primeiro representando o número do aluno e o segundo
representando a sua altura em centímetros. Encontre o aluno mais alto e o mais baixo. Mostre o número do aluno mais alto
e o número do aluno mais baixo, junto com suas alturas. """

alunos = []
alturas = []
cont = alto = baixo = 0

def valida(x):
    while x < 0 or x in alunos:
        x = int(input('Entrada inválida ou repetida. Digite novamente [0 para sair]: '))
    return x

while True:
    cont += 1

    num = int(input('Número do aluno [0 para sair]: '))
    if num < 0 or num in alunos:
        num = valida(num)
    if num == 0:
        break
    alunos.append(num)

    alt = int(input('Altura do aluno em cm: '))
    if alt < 0:
        alt = valida(alt)
    if cont == 1:
        alto = baixo = alt
    else:
        if alt > alto:
            alto = alt
        if alt < baixo:
            baixo = alt
    alturas.append(alt)

juncao = list(zip(alunos, alturas))

print(f'\nO aluno mais ALTO possui {alto} cm de altura e é o de número {alunos[alturas.index(alto)]} no cadastro.')
print(f'O aluno mais BAIXO possui {baixo} cm de altura e é o de número {alunos[alturas.index(baixo)]} no cadastro.')



