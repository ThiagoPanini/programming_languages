# Laço infinito e condição break

n = s = 0
while True:  # Laço infinito
    n = int(input('Digite um número: '))
    if n == 999:    # Se a flag de saída for atendida
        break  # Sai do laço
    s += n  # Se a flag for cumprida, ela não entra na soma pois o break está antes.
print('A soma vale {}.'.format(s))
