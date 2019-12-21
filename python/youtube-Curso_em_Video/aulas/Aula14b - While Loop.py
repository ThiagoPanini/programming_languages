# Aula 14 - Condicionando entrada para saída do laço

n = 1
while n != 0: # Flag = Condição de parada. Se digitado 0, sai do laço.
    n = int(input('Digite um número: '))
print('Fim')

r = 'S'
while r == 'S':
    n = int(input('Digite um valor: '))
    r = input('Deseja continuar? [S/N]: ').upper()
print('Fim')