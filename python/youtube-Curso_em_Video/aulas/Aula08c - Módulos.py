#Importando Módulos e trabalhando com as Bibliotecas

#Importando biblioteca que não é built-in -> aba Python Package Index do site Python.org
#Biblioteca 'emoji'

import emoji
from random import choice

print(emoji.emojize('Olá, mundo! :earth_americas:', use_aliases=True))

lista = [emoji.emojize(':earth_americas:', use_aliases=True), 'Outro']

print(choice(lista))

