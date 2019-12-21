#Desafio 41
#A confederação de natação precisa de um programa que leia o ano de nascimento de um atleta e mostre sua categoria, de acordo com a idade:
# - Até 9 anos: MIRIM
# - Até 14 anos: INFANTIL
# - Até 19 anos: JUNIOR
# - Até 20 anos: SENIOR
# - Acima: MASTER

from datetime import date

nasc = int(input('Digite o ano de nascimento do atleta: '))
ano = date.today().year
idade = ano-nasc
print('O atleta possui {} anos.'.format(idade))

if idade <= 9:
    print('Se enquadra na categoria \033[1;30;41mMIRIM.\033[m')
elif 9 < idade <= 14:
    print('Se enquadra na categoria \033[1;30;42mINFANTIL.\033[m')
elif 14 < idade <= 19:
    print('Se enquadra na categoria \033[1;30;43mJUNIOR.\033[m')
elif 19 < idade <= 20:
    print('Se enquadra na categoria \033[1;30;44mSENIOR.\033[m')
else:
    print('Se enquadra na categoria \033[1;30;45mMASTER.\033[m')
