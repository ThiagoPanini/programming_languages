"""4.8 Faça um Programa que peça a idade e a altura de 5 pessoas, armazene cada informação no seu respectivo vetor.
Imprima a idade e a altura na ordem inversa a ordem lida."""

idades = []
alturas = []

for i in range(5):
    idades.append(int(input('Digite sua idade: ')))
    alturas.append(float(input('Agora sua altura em metros: ')))
idades.reverse()
alturas.reverse()
print(f'Idades em ordem reversa: {idades}')
print(f'Alturas em ordem reverse: {alturas}')