"""3.37 Uma academia deseja fazer um senso entre seus clientes para descobrir o mais alto, o mais baixo, a mais gordo e
o mais magro, para isto você deve fazer um programa que pergunte a cada um dos clientes da academia seu código, sua
altura e seu peso. O final da digitação de dados deve ser dada quando o usuário digitar 0 (zero) no campo código.
Ao encerrar o programa também deve ser informados os códigos e valores do clente mais alto, do mais baixo, do mais gordo
 e do mais magro, além da média das alturas e dos pesos dos cliente"""

def valida(x):
    while x < 0:
        x = float('Entrada inválida. Digite novamente.')
    return x

print('-'*20)
print('ACADEMIA PYTHON-GYM')
print('-'*20)

cadastros = []
pesos = []
alturas = []
maiorpeso = menorpeso = maioraltura = menoraltura = 0
i = 1

while True:
    cod = int(input('\nDigite seu código de cadastro [0 PARA SAIR]:  '))
    while cod < 0:
        cod = int(input('Código inválido. Digite um número maior que 0 [0 PARA SAIR]: '))
    if cod == 0:
        break
    while cod in cadastros:
        cod = int(input('Código já cadastrado. Digite novamente: '))
    cadastros.append(cod)

    altura = float(input('Digite sua altura (em metros): '))
    while altura < 0:
        altura = float(input('Altura inválida. Digite um número maior que 0 (em metros): '))
    alturas.append(altura)
    if i == 1:
        maioraltura = menoraltura = altura
    else:
        if altura > maioraltura:
            maioraltura = altura
        if altura < menoraltura:
            menoraltura = altura

    peso = float(input('Digite seu peso (em Kg): '))
    while peso < 0:
        peso = float(input('Peso inválido. Digite um número maios que 0 (em kg): '))
    pesos.append(peso)
    if i == 1:
        maiorpeso = menorpeso = peso
    else:
        if peso > maiorpeso:
            maiorpeso = peso
        if peso < menorpeso:
            menorpeso = peso

    i += 1

"""print(cadastros)
print(alturas)
print(pesos)
print('\n')"""
print('TABELA DE CADASTRO:')
geral = list(zip(cadastros, alturas, pesos))
print(geral)
print(f'\nMaior altura: {maioraltura} - Código cadastrado: {cadastros[alturas.index(maioraltura)]}\n'
      f'Menor altura: {menoraltura} - Código cadastrado: {cadastros[alturas.index(menoraltura)]}\n' 
      f'Maior peso: {maiorpeso} - Código cadastrado: {cadastros[pesos.index(maiorpeso)]}\n'
      f'Menor peso: {menorpeso} - Código cadadstrado: {cadastros[pesos.index(menorpeso)]}\n')
