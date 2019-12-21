# Desafio 57
# Faça um programa que leia o sexo de uma pessoa, mas só aceite os valores 'M' ou 'F'.
# Caso esteja errado, peça a digitação novamente até ter um valor correto.

s = ''
contm = 0
contf = 0
while True:
    s = input('Digite o sexo da pessoa [M/F]: ').strip().upper()
    while s != 'M' and s != 'F':
        s = input('Valor inválido. Digite "M" para Masculino e "F" para Feminino').upper()
    if s == 'M':
        contm += 1
    if s == 'F':
        contf += 1
    print('Contagem: {} Homens e {} Mulheres'.format(contm, contf))

# Uma outra forma de tratar seria:

#s = input('Informe seu sexo: [M/F] ').strip().upper()[0] #primeira maiuscula apenas
# while s not in 'MF' (ou 'MmFf')
    #s = input('Dados inválidos. Informe seu sexo novamente: ').strip().upper()[0]
#print('Sexo {} registrado com sucesso').format(s)

#Resolução Guanabara funciona apenas uma vez.
    #s = input('


