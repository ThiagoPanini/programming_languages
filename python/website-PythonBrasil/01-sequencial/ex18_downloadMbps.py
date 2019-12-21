"""1.18 Faça um programa que peça o tamanho de um arquivo para download (em MB) e a velocidade de um link de Internet
(em Mbps), calcule e informe o tempo aproximado de download do arquivo usando este link (em minutos)."""

file = float(input('Tamanho do arquivo em MB: '))
speed = float(input('Velocidade de download em Mbps: '))
tempo = (file/speed)/60
print(f'Tempo em segundos: {tempo*60:.2f}. Tempo em minutos: {tempo:.2f}')