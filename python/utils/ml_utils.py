"""
    Script criado para consolidar funções úteis utilizadas no treinamento dos mais variados modelos de
    Machine Learning.
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from utils.viz_utils import *


"""
--------------------------------------------
-------- 1. MODELOS DE AGRUPAMENTO ---------
--------------------------------------------
"""


# Função para plotagem do Método do Cotovelo (agrupamento)
def elbow_method_kmeans(df, K_min, K_max, figsize=(10, 5)):
    """
    Etapas:
        1. treinar diferentes modelos KMeans pra cada cluster no range definido
        2. plotar método do cotovelo (elbow método) baseado na distância euclidiana

    Argumentos:
        df -- dados já filtrados com as colunas alvo de análise [pandas.DataFrame]
        K_min -- índice mínimo de análise dos clusters [int]
        K_max -- índice máximo de análise dos clusters [int]
        figsize -- dimensões da figura de plotagem [tupla]

    Retorno:
        None
    """

    # Treinando algoritmo KMeans para diferentes clusters
    square_dist = []
    for k in range(K_min, K_max):
        km = KMeans(n_clusters=k)
        km.fit(df)
        square_dist.append(km.inertia_)

    # Plotando análise Elbow
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=range(K_min, K_max), y=square_dist, color='cornflowerblue', marker='o')

    # Customizando gráfico
    format_spines(ax, right_border=False)
    ax.set_title('Elbow Method - Modelo KMeans', size=14, color='dimgrey')
    ax.set_xlabel('Número de Clusters')
    ax.set_ylabel('Distância Euclidiana')
    plt.show()


# Função para plotagem do resultado do algoritmo KMeans treinado
def plot_kmeans_clusters(df, y_kmeans, centers, figsize=(14, 7), cmap='viridis'):
    """
    Etapas:
        1. retorno de parâmetros de plotagem
        3. plotagem de clusters já preditos

    Argumentos:
        df -- conjunto de dados utilizados no algoritmo KMeans [pandas.DataFrame]
        y_kmeans -- predições do modelo (cluster ao qual o registro se refere) [np.array]
        centers -- centróides de cada cluster [np.array]
        figsize -- dimensões da figura de plotagem [tupla]
        cmap -- mapeamento colorimétrico da plotagem [string]

    Retorno:
        None
    """

    # Retornando valores e definindo layout
    variaveis = df.columns
    X = df.values
    sns.set(style='white', palette='muted', color_codes=True)

    # Plotando gráfico de dispersão
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap=cmap)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    # Customizando gráfico
    ax.set_title(f'K_Means aplicado entre {variaveis[0].upper()} e {variaveis[1].upper()}', size=14, color='dimgrey')
    format_spines(ax, right_border=False)
    ax.set_xlabel(variaveis[0])
    ax.set_ylabel(variaveis[1])
    plt.show()