"""3.16 A série de Fibonacci é formada pela seqüência 0,1,1,2,3,5,8,13,21,34,55,... Faça um programa que gere a série
até que o valor seja maior que 500. """

print('-'*20)
print('FIBONACCI até 500.')
print('-'*20)

anterior = 1
atual = 1
proximo = 2
i = 0

while(anterior <= 500):
    print(anterior, end=' -> ')
    anterior = atual
    atual = proximo
    proximo = anterior + atual
    i += 1
print(f'Foram mostrados {i} termos da sequência.')