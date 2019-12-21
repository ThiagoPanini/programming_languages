"""4.12 Foram anotadas as idades e alturas de 30 (10) alunos. Faça um Programa que determine quantos alunos com mais de
13 anos possuem altura inferior à média de altura desses alunos."""

idades = [12, 13, 13, 13, 14, 14, 12, 13, 15, 16]
alturas = [1.50, 1.67, 1.63, 1.54, 1.76, 1.54, 1.56, 1.67, 1.60, 1.59]

mediaalt = sum(alturas)/len(alturas)
cont = 0
for idade in idades:
    if idade > 13:
        if alturas[idades.index(idade)] < mediaalt:
            cont += 1

print(f'A média de alturas é {mediaalt:.2f} e há {cont} alunos com mais de 13 anos e altura abaixo da média.')