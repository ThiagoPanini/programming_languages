"""4.9 Faça um Programa que leia um vetor A com 10 números inteiros, calcule e mostre a soma dos quadrados dos elementos
 do vetor. """

lista = [int(input('Insira um elemento da lista: '))for x in range(10)]


def somaquadrados():
    soma = 0
    quadrados = []
    global lista
    for item in lista:
        soma += item**2
        quadrados.append(item**2)
    return soma, quadrados


soma, quadrados = somaquadrados()
print(f'A lista de quadrados é: {quadrados}')
print(f'A soma dos elementos ao quadrado é: {soma}')



