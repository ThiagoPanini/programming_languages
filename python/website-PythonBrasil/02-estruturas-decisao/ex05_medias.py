"""2.5 Faça um programa para a leitura de duas notas parciais de um aluno.
O programa deve calcular a média alcançada por aluno e apresentar:
   * A mensagem "Aprovado", se a média alcançada for maior ou igual a sete;
   * A mensagem "Reprovado", se a média for menor do que sete;
   * A mensagem "Aprovado com Distinção", se a média for igual a dez."""

n1 = float(input('Primeira nota: '))
n2 = float(input('Segunda nota: '))
media = (n1+n2)/2

if media >= 7 and media < 10:
    print(f'A média do aluno foi de {media:.2f} e ele está APROVADO.')
elif media < 7:
    print(f'A média do aluno foi de {media:.2f} e ele está REPROVADO.')
elif media == 10:
    print(f'Média 10! Aprovado com distinção!')