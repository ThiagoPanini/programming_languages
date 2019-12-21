#Manipulando strings

frase = 'Curso em Vídeo Python'
print(frase)

#Fatiamento
print(frase[3]) #Retorna: 's'
print(frase[3:13]) #Retorna: 'so em Víde'
print(frase[:13]) #Retorna: 'Curso em Víde'
print(frase[1:15]) #Retorna: 'urso em Vídeo'
print(frase[1:15:2]) #Retorna: 'us mVdo'
print(frase[::2]) #Retorna: 'Croe íe yhn'

#Bonus: Print em frases longas (""")
print("""Old at heart but I'm only twenty eight
And I'm much too young
To let love break my heart
Young at heart but it's getting much too late
To find ourselves so far apart""") #3 aspas para textos longos

#Outras transformações
print(frase.count('o')) #Resultado: 3
print(frase.count('O')) #Resultado: 0 (diferenciação maiúsculo)
print(frase.upper().count('O')) #Resultado: 3 (upper deixou tudo maiúsculo)
print(len(frase)) #Resultado: 21 (será MUITO utilizado)
print(frase.replace('Python', 'Android')) #Resultado: 'Curso em Vídeo Android'
#frase[0] = 'J' Resultado: Erro. String é imutável. Nem o replace muda
print(frase)
frase = frase.replace('Python', 'Android') #Agora sim ele atribui a mudança à string frase
print('Curso' in frase) #Resultado: True (existe 'Curso' em 'frase')
print(frase.find('Curso')) #Resultado: 0 ('Curso' começa no caractere 0)
print(frase.find('Vídeo')) #Resultado: 9 ('Vídeo' começa no caractere 9)
print(frase.find('video')) #Resultado: -1 (Diferenciação de minúsculas e acento)
print(frase.lower().find('vídeo')) #Resultado: 9 (lower ajudou)

#Split
print(frase.split()) #Resultado: ['Curso', 'em', 'Vídeo', 'Python']
dividido = frase.split()
print(dividido[0]) #Resultado: 'Curso' (Item 0 da lista dividido)
print(dividido[2][3]) #Resultado: 'e' (Pega o item [2] da lista dividido e me mostre o caractere 3 dele)