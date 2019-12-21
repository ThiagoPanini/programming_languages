"""Crie um programa que tenha uma tupla totalmente preenchida com uma contagem
por extensão de zero até vinte.
Seu programa deverá ler um número pelo teclado (entre 0 e 20) e mostrá-lo por extenso"""

tupla = ("zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez",
         "onze", "doze", "treze", "catorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove", "vinte")

num = int(input('Digite um número entre 0 e 20: '))
while num not in range(21):
    num = int(input("Número inválido. Digite entre 0 e 20: "))

print("Número selecionado por extenso: {}.".format(tupla[num]))