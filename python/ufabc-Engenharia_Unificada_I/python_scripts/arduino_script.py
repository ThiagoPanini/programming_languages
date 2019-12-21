import pandas as pd
import serial
import time


def show_menu():
    """Cria menu de seleção para o usuário"""
    menu = 0
    while menu not in (range(1, 6)):
        print('[1] - Ler um Cartão')
        print('[2] - Status do Estacionamento')
        print('[3] - Quantidade de Vagas Disponíveis')
        print('[4] - Quantidade de Vagas Utilizadas')
        print('[5] - Encerrar programa')

        # Valida possíveis erros
        try:
            menu = int(input(''))
        except:
            print('Opção inválida. Tente novamente!')
            pass
        if menu not in range(1, 6):
            print('Opção inválida. Tente novamente!')
            pass

    return menu


def main():
    """Definição do programa principal"""

    # Lendo banco de dados de acessos
    pathfile = r'D:\Users\thiagoPanini\programming-lessons\python\ufabc-engenharia-unificada-I\tags_acesso.txt'
    acessos = pd.read_csv(pathfile, sep=';')
    tags = list(acessos['tag'])
    acessos = acessos.set_index('tag')
    acessos['status'] = acessos.apply(lambda _: 'saída', axis=1)

    # Definições iniciais
    vagas = len(acessos)
    ser = serial.Serial('COM5', 9600)

    # Retornando opção selecionada
    while True:
        menu = show_menu()
        # Leitura de cartão
        if menu == 1:
            print('--- APROXIME O CARTÃO ---\n')
            try:
                tag = ser.readline().rstrip().decode("utf-8").strip()
                if tag in tags:
                    nome = str(acessos.loc[tag, ['nome']].values[0])
                    ra = acessos.loc[tag, ['id']].values[0]
                    tipo_acesso = str(acessos.loc[tag, ['tipo_de_acesso']].values[0]).capitalize()
                    print(f'{tipo_acesso} {nome}, ID: {ra}')
                    print('Liberado!\n')
                    if acessos.loc[tag, ['status']].values[0] == 'saída':
                        acessos.loc[tag, ['status']] = 'entrada'
                    else:
                        acessos.loc[tag, ['status']] = 'saída'
                    pass
                else:
                    pass
            except:
                print('Data could not be read\n')
                pass

        # Status do estacionamento
        if menu == 2:
            print('--- STATUS DO ESTACIONAMENTO ---')
            print()
            print(acessos)
            print()

        # Quantidade de Vagas Disponíveis
        if menu == 3:
            vagas_disp = sum(acessos['status'].values == 'saída')
            print(f'Total de vagas: {vagas}')
            print(f'Vagas disponíveis: {vagas_disp}\n')

        # Quantidade de Vagas Utilizadas
        if menu == 4:
            vagas_disp = sum(acessos['status'].values == 'entrada')
            print(f'Total de vagas: {vagas}')
            print(f'Vagas utilizadas: {vagas_disp}\n')

        # Encerrando programa
        if menu == 5:
            ser.close()
            break


# Programa Principal
if __name__== "__main__":
    main()