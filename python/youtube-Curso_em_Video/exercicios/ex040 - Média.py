#Desafio 40
#Crie um programa que leia duas notas de um aluno e calcule sua média, mostrando uma mensagem no final, de acordo com a média atingida:
# - Média abaixo de 5.0: REPROVADAO
# - Média entre 5.0 e 6.9: RECUPERAÇÃO
# - Média 7.0 ou superior: APROVADO

n1 = float(input('Digite a primeira nota do aluno: '))
n2 = float(input('Digite a segunda nota do aluno: '))
media = (n1+n2)/2

if media < 5.0:
    print('Sua média foi de {:.2f}. Aluno \033[4;36mREPROVADO\033[m.'.format(media))
elif media >= 7.0:
    print('Sua média foi de {:.2f}. Aluno \033[4;34mAPROVADO\033[m!'.format(media))
else:
    print('Sua média foi de {:.2f}. Aluno de \033[4;35mRECUPERAÇÃO\033[m!'.format(media))