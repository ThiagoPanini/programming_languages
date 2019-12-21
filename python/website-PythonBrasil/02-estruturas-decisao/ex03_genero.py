"""2.3 Faça um Programa que verifique se uma letra digitada é "F" ou "M".
Conforme a letra escrever: F - Feminino, M - Masculino, Sexo Inválido. """

sexo = input('Gênero [M/F]: ')

if sexo in 'Mm':
    print('Sexo MASCULINO.')
elif sexo in 'Ff':
    print('Sexo FEMININO.')
else:
    print('Valor inválido.')