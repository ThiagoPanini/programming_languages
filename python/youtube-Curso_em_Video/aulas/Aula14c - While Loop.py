# Aula 14 - Tratando Flags

n = 1
par = impar = 0
while n != 0: # Com essa condição, o 0 entra na conta na última verificação, contando como PAR.
    n = int(input('Digite um valor: '))
    if n != 0: # Agora sim. As condições abaixo funcionam somente se o número for diferente de 0. Se for 0 já sai direto.
        if n % 2 == 0:
            par += 1
        else:
            impar += 1

print('Você digitou {} números pares e {} números ímpares'.format(par, impar))