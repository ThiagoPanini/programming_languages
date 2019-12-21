"""3.27 Faça um programa que calcule o número médio de alunos por turma.
Para isto, peça a quantidade de turmas e a quantidade de alunos para cada turma.
As turmas não podem ter mais de 40 alunos."""

soma = 0

turmas = int(input('Quantidade de turmas: '))
while turmas < 0:
    turmas = int(input('Número inválido. Insira a quantidade de turmas: '))

for c in range(turmas):
    alunos = int(input(f'Quantidade de alunos na turma {c+1} (máx 40): '))
    while alunos > 40:
        alunos = int(input(f'Número máximo de alunos excedido. Digite novamente para a turma {c+1} (máx 40): '))
    soma += alunos
print(f'Total de alunos: {soma}.\nTotal de turmas: {turmas}.')
print(f'Média de alunos por turma: {soma/turmas:.0f}.')
