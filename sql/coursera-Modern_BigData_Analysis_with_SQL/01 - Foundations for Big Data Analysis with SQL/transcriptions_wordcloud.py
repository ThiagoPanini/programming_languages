"""
    Script desenvolvido para criar wordClouds a partir das transcrições de áudio dos vídeos.
O objetivo dessa implementação é facilitar o aprendizado e propor uma forma fácil e eficiente de assimilar o
conteúdo de vídeos por palavras-chave.
"""

"""
------------------------------------------------------------------------------------------------
-------------------------------- IMPORTANDO BIBLIOTECAS ----------------------------------------
------------------------------------------------------------------------------------------------
"""
import os
import sys
import re
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud, STOPWORDS
except:
    print('Biblioteca wordcloud inexistente. Realizando download via pip\n')
    try:
        os.system('pip install wordcloud')
        from wordcloud import WordCloud, STOPWORDS
        print('Download realizado com sucesso.\n')
    except:
        print(f'Não foi possível realizar o download via pip. Encerrando script.')
        sys.exit()

"""
------------------------------------------------------------------------------------------------
---------------------------------- DEFININDO CLASSES -------------------------------------------
------------------------------------------------------------------------------------------------
"""


class CloudGenerator:

    def __init__(self, raw_path):
        self.raw_path = raw_path

    def collect_transcriptions(self, weeks):
        """
        Função responsável por navegar entre o diretório fornecido e retornar apenas arquivos referentes as
        transcrições de áudio em formato txt

        :return dict_transcriptions: dicionário contendo as transcrições (valores) por semana (chaves)
        """
        # Navegando pelo diretório raw
        os.chdir(self.raw_path)
        transcriptions = []
        dict_transcriptions = {}

        for dir_path, dirs, files in os.walk(self.raw_path):
            temp_transcript = [f for f in files if 'Transcriptions' in f]
            transcriptions += temp_transcript

        # Salvando transcrições por semana
        for i in range(1, weeks + 1):
            dict_transcriptions['Semana ' + str(i)] = [t for t in transcriptions if t[2] == str(i)]

        return dict_transcriptions

    def generate_wordcloud(self, transcriptions):
        """
        Função responsável por ler o conteúdo das transcrições, consolidá-lo por semana e gerar uma wordCloud
        pra todos os arquivos lidos dentro daquela respectiva semana

        :param transcriptions: lista com nome de arquivos contendo transcrições de áudio
        :return: imagens png salvas no mesmo diretório dos arquivos lidos
        """
        # Iterando por dicionário de transcrições
        for week_ref, transc in transcriptions.items():
            # Verifica se há transcrições pra semana em questão
            if len(transc) == 0:
                break
            words = ''
            week_path = f'\{week_ref}'
            for file_name in transc:
                # Definindo caminho
                file_path = self.raw_path + week_path + f'\{file_name}'

                # Adicionando todas as transcrições da semana em uma string única
                with open(file_path, 'r') as f:
                    content = ' '.join(f.readlines())
                    words += content

            # Preparando string (removendo urls)
            words = re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
                           'link', words)

            # Removendo números
            words = re.sub('\d+(?:\.\d*(?:[eE]\d+))?', 'number', words)

            # Caracteres especiais
            words = re.sub(r'\W', ' ', words)

            # Espaços em branco
            words = re.sub(r'\s+', ' ', words)

            # Lower case
            words = words.lower()

            # WordCloud
            stopwords = set(list(STOPWORDS) + ['data', 'll'])
            wordcloud = WordCloud(stopwords=stopwords, background_color="white", collocations=False).generate(words)

            plt.figure(figsize=(15, 15))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(week_path[1:] + ' WordCloud', size=30, pad=20)
            word_cloud_figname = self.raw_path + week_path + week_path.replace(' ', '_') + '_WordCloud.png'
            plt.savefig(word_cloud_figname, format='png', dpi=300)


"""
------------------------------------------------------------------------------------------------
--------------------------------- PROGRAMA PRINCIPAL -------------------------------------------
------------------------------------------------------------------------------------------------
"""


def main():

    # Definindo variáveis
    study_path = r'D:\Users\thiagoPanini\programming-lessons\sql\coursera - Modern Big Data Analysis with SQL'
    first_course_path = study_path + r'\01 - Foundations for Big Data Analysis with SQL'
    second_course_path = study_path + r'\02 - Analyzing Big Data with SQL'
    weeks_first_course = 5
    weeks_second_course = 6
    specialization_stuff = {
        'first_course': [first_course_path, weeks_first_course],
        'second_course': [second_course_path, weeks_second_course]
    }

    # Coletando transcrições por semana e gerando wordClouds pra cada curso
    for course, stuff in specialization_stuff.items():
        raw_path = stuff[0]
        weeks = stuff[1]
        wc_generator = CloudGenerator(raw_path=raw_path)
        transcriptions = wc_generator.collect_transcriptions(weeks=weeks)
        # Gerando WordClouds
        wc_generator.generate_wordcloud(transcriptions=transcriptions)


if __name__ == '__main__':
    main()