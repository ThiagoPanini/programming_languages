#Testes de métodos

n = float(input('Digite um valor: '))
print(n)
#Tipo float converte um número inteiro em ponto flutuante (Ex: 4 = 4.0)

n1 = bool(input('Digite algo: '))
print(n1)
#Tipo booleano verifica se há algo digitado e retorna True. Caso aperte Enter direto, resultado é False

#_________TESTANDO MÉTODOS___________

n2 = input('Digite algo: ')
print(n2.isnumeric()) #Teste se o valor digitado realmente é um número

n3 = input('Digite algo: ')
print(n3.isalpha()) #Testa se o valor digitado é letra(palavras)

#Lembrando que o tipo do 'input' é sempre uma String e, caso converta ele pra outro tipo, a história é outra...