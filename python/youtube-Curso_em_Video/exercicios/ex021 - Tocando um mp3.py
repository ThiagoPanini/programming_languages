#Desafio 21 - Faça um programa que abra e reproduza o áudio de um arquivo mp3

import pygame #Instalar packaging pygame -> não é built-in
pygame.init() #Inicia o pygame
pygame.mixer.music.load('ex021.mp3')
pygame.mixer.music.play()
pygame.event.wait()