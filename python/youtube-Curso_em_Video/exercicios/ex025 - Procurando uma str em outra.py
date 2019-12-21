#Desafio 25
#Crie um programa que leia o nome de uma pessoa e diga se ela tem "Silva" no nome.

nome = input('Digite o nome: ').strip().upper().split()
#função strip evita que usuário coloque espaços adicionais e indesejados
#função upper engloba todas as variações maiúsculas e minúsculas de "Silva"
#função split separa os nomes em listas para evitar resultados VERDADEIROS em variações de "Silva" (Ex: Silvana)

print('O nome digitado possui "Silva"? {}'.format('SILVA' in nome))
