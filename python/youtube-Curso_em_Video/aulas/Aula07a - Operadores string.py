#Operadores com String

nome = input('Qual o seu nome? ')
print('Prazer em te conhecer, {:20}!'.format(nome)) #Printa o nome em 20 caracteres

nome = input('Qual o seu nome? ')
print('Prazer em te conhecer, {:>20}!'.format(nome)) #Printa o nome em 20 caracteres e alinha a direita

nome = input('Qual o seu nome? ')
print('Prazer em te conhecer, {:<20}!'.format(nome)) #Printa o nome em 20 caracteres e alinha a esquerda

nome = input('Qual o seu nome? ')
print('Prazer em te conhecer, {:^20}!'.format(nome)) #Printa o nome em 20 caracteres e centraliza

nome = input('Qual o seu nome? ')
print('Prazer em te conhecer, {:=^20}!'.format(nome))#Printa o nome em 20 caracteres, centraliza e coloca o símbolo '=' em volta
