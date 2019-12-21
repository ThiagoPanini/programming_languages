"""2.27 Faça um programa que faça 5 perguntas para uma pessoa sobre um crime. As perguntas são:
   a. "Telefonou para a vítima?"   b. "Esteve no local do crime?"   c. "Mora perto da vítima?"
   d. "Devia para a vítima?"   e. "Já trabalhou com a vítima?"
 2.28   O programa deve no final emitir uma classificação sobre a participação da pessoa no crime.
 Se a pessoa responder positivamente a 2 questões ela deve ser classificada como "Suspeita", entre 3 e 4 como "Cúmplice"
  e 5 como "Assassino". Caso contrário, ele será classificado como "Inocente". """

print('='*20)
print('INVESTIGAÇÃO PYTHON!')
print('='*20)
lista = []

r = input('Pergunta número 1: "Telefonou para a vítima?" [SIM / NÃO] ').strip().upper()[0]
lista.append(r)
r = input('Pergunta número 2: "Esteve no local do crime?" [SIM / NÃO] ').strip().upper()[0]
lista.append(r)
r = input('Pergunta número 3: "Mora perto da vítima?" [SIM / NÃO] ').strip().upper()[0]
lista.append(r)
r = input('Pergunta número 4: "Devia para a vítima?" [SIM / NÃO] ').strip().upper()[0]
lista.append(r)
r = input('Pergunta número 5: "Já trabalhou com a vítima?" [SIM / NÃO] ').strip().upper()[0]
lista.append(r)

if lista.count('S') == 2:
    print('Sujeito considerado SUSPEITO.')
elif lista.count('S') == 3 or lista.count('S') == 4:
    print('Sujeito considerado CÚMPLICE.')
elif lista.count('S') == 5:
    print('Sujeito considerado CULPADO!')
else:
    print('Sujeito considerado INOCENTE.')

print('Fim do caso.')