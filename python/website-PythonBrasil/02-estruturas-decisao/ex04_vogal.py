"""2.4 Faça um Programa que verifique se uma letra digitada é vogal ou consoante. """

vogal = ['A', 'E', 'I', 'O', 'U']
letra = input('Insira uma letra: ').strip().upper()

if letra in vogal:
    print(f'"{letra}" é uma vogal.')
else:
    print(f'"{letra}" é uma consoante.')