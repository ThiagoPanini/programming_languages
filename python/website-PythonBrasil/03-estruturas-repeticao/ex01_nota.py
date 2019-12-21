"""3.1 Faça um programa que peça uma nota, entre zero e dez. Mostre uma mensagem caso o valor seja inválido e
continue pedindo até que o usuário informe um valor válido. """

nota = -1
while nota not in range(0,11):
    nota = float(input('Digite uma nota entre 0 e 10: '))
print('Ou então de uma outra forma')
nota = float(input('Digite uma nota entre 0 e 10: '))
while nota not in range(0,11):
    nota = float(input('Nota inválida. Escolha entre 0 e 10: '))
print('Ok')