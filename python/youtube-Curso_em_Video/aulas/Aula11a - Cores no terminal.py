#Sistemas de Cores no Terminal

print('\033[1;31;43mOlá, Mundo!') #Negrito, letra vermelha e fundo amarelo (até o final)
print('\033[1;31;43mOlá, Mundo!\033[m') #Tira as formatações no final da frase
print('\033[4;30;45mOlá, Mundo!\033[m')
print('\033[7;30mOlá, Mundo!\033[m') #7 = Inverte letra com fundo (letra branca e fundo preto invertido)
print('\033[7;33;44mOlá, Mundo!\033[m') #Inverte letra amarela e fundo azul