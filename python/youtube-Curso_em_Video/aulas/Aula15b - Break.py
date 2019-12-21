# fstrings = uma nova forma de simplificar a forma de escrever com o print

n = s = 0
while True:  # Laço infinito
    n = int(input('Digite um número: '))
    if n == 999:
        break
    s += n
print(f'A soma vale {s}.')  # Coloca um 'f' antes das aspas e escreve a variável dentro das chaves.
#print('A soma vale {}.'.format(s))