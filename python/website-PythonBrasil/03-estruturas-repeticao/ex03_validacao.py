"""3.3 Faça um programa que leia e valide as seguintes informações:
a. Nome: maior que 3 caracteres;
b. Idade: entre 0 e 150;
c. Salário: maior que zero;
d. Sexo: 'f' ou 'm';
e. Estado Civil: 's', 'c', 'v', 'd';"""

print('-'*8)
print('CADASTRO')
print('-'*8)
cadastro = []
atributos = ('Nome', 'Idade', 'Salário', 'Sexo', 'Estado Civil')

nome = input('Nome: ')
while len(nome) <= 3:
    nome = input('Nome deve ser maior que três caracteres. Digite novamente: ')
cadastro.append(nome)

idade = int(input('Idade: '))
while idade not in range(0, 151):
    idade = int(input('Idade deve ser entre 0 e 150. Digite novamente: '))
cadastro.append(idade)

salario = float(input('Salário: R$'))
while salario <= 0:
    salario = float(input('Salário deve ser maior que R$0,00. Digite novamente: R$'))
cadastro.append(salario)

sexo = input('Sexo: [M/F] ').strip().upper()[0]
while sexo not in 'MmFf':
    sexo = input('Valor inválido. Digite [ M ] ou [ F ] para o sexo: ')
cadastro.append(sexo)

civil = input('Estado civil: \n'
              '[ S ] Solteiro\n'
              '[ C ] Casado\n'
              '[ V ] Viúvo\n'
              '[ D ] Divorciado\n').strip().upper()[0]
while civil not in 'SCVD':
    civil = input('Valor inválido. Digite novamente:\n'
              '[ S ] Solteiro\n'
              '[ C ] Casado\n'
              '[ V ] Viúvo\n'
              '[ D ] Divorciado\n').strip().upper()[0]
cadastro.append(civil)

for item in atributos:
    print('{:^15}'.format(item), end='')
print('\n')
for item in cadastro:
    print('{:^15}'.format(item), end='')
print('\n\nFim.')