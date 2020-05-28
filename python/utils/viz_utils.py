"""
    Script criado para consolidar funções úteis utilizadas na plotagem e personalização de gráficos
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
--------------------------------------------
---------- 1. FORMATAÇÃO DE EIXOS ----------
--------------------------------------------
"""


# Formatando eixos do matplotlib
def format_spines(ax, right_border=True):
    """
    This function sets up borders from an axis and personalize colors

    Input:
        Axis and a flag for deciding or not to plot the right border
    Returns:
        Plot configuration
    """
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')


"""
--------------------------------------------
---------- 2. PLOTAGENS GRÁFICAS -----------
--------------------------------------------
"""


# Função para plotagem de boxenplot (https://stackoverflow.com/questions/52403381/how-boxen-plot-is-different-from-box-plot)
def boxenplot(data, label, features, n_rows, n_cols, figsize=(16, 8), palette='viridis'):
    """
    Etapas:
        1. criação de figura de acordo com as especificações dos argumentos
        2. laço para plotagem de boxplot por eixo
        3. formatação gráfica
        4. validação de eixos excedentes

    Argumentos:
        data -- base de dados para plotagem [pandas.DataFrame]
        label -- variável resposta contida na base [string]
        features -- conjunto de colunas a serem avaliadas [list]
        n_rows, n_cols -- especificações da figura do matplotlib [int]
        palette -- paleta de cores [string]

    Retorno:
        None
    """

    # Validando parâmetros de figura inseridos
    n_features = len(features)
    if (n_rows == 1) & (n_cols < n_features) | (n_cols == 1) & (n_rows < n_features):
        print(f'Com a combinação de n_rows ({n_rows}) e n_cols ({n_cols}) não será possível plotar ' \
              f'todas as features ({n_features})')
        return None

    # Criando figura para plotagem fráfica
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Definindo índices para percorrer os eixos
    i, j = (0, 0)
    for feat in features:
        # Tratando possíveis erros nas definições dos eixos
        try:
            ax = axs[i][j]
        except IndexError as e:
            print(f'Número de features ({n_features}) excede a quantidade de eixos ' \
                  f'definidos por n_rows ({n_rows}) e n_cols ({n_cols})! \n{e.args}')
            return None
        except TypeError as e:
            try:
                ax = axs[j]
            except IndexError as e:
                print(f'Número de features ({n_features}) excede a quantidade de eixos ' \
                      f'definidos por n_rows ({n_rows}) e n_cols ({n_cols})! \n{e.args}')
                return None

        # Plotando gráfico
        sns.boxenplot(x=data[label], y=data[feat], ax=ax, palette=palette)

        # Formatando gráfico
        format_spines(ax, right_border=False)
        ax.set_title(f'Feature: {feat.upper()}', size=14, color='dimgrey')
        plt.tight_layout()

        # Incrementando
        j += 1
        if j == n_cols:
            j = 0
            i += 1

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(n_rows * n_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= n_features:
            try:
                axs[i][j].axis('off')
            except TypeError as e:
                axs[j].axis('off')

        # Incrementando
        j += 1
        if j == n_cols:
            j = 0
            i += 1


# Distplot para comparação de densidade das features baseadas na variável target
def distplot(data, target_column, target_names, features, color_list, n_rows, n_cols, figsize=(16, 8), hist=False):
    """
    Etapas:
        1. criação de figura de acordo com as especificações dos argumentos
        2. laço para plotagem de boxplot por eixo
        3. formatação gráfica
        4. validação de eixos excedentes

    Argumentos:
        data -- base de dados para plotagem [pandas.DataFrame]
        target_column -- variável resposta contida na base [string]
        target_names -- rótulos dados à variável resposta [list]
        features -- conjunto de colunas a serem avaliadas [list]
        color_list -- cores para identificação de cada classe nos gráficos [list]
        n_rows, n_cols -- especificações da figura do matplotlib [int]
        figsize -- dimensões da plotagem [tupla]
        hist -- indicador de plotagem das faixas do histograma [bool]

    Retorno:
        None
    """

    # Separando variável target
    unique_vals = data[target_column].unique()
    targets = [data[data[target_column] == val] for val in unique_vals]

    # Definindo variáveis de controle
    i, j, color_idx = (0, 0, 0)
    n_features = len(features)

    # Validando configuração de linhas e colunas
    if (n_rows == 1) & (n_cols < n_features) | (n_cols == 1) & (n_rows < n_features):
        print(f'Com a combinação de n_rows ({n_rows}) e n_cols ({n_cols}) não será possível plotar ' \
              f'todas as features ({n_features})')
        return None

    # Plotando gráficos
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    # Percorrendo por cada uma das features
    for feat in features:
        # Tratando possíveis erros nas definições dos eixos
        try:
            ax = axs[i][j]
        except IndexError as e:
            print(f'Número de features ({n_features}) excede a quantidade de eixos ' \
                  f'definidos por n_rows ({n_rows}) e n_cols ({n_cols})! \nSerão mostradas apenas {n_rows * n_cols} features.')
            return None
        except TypeError as e:
            try:
                ax = axs[j]
            except IndexError as e:
                print(f'Número de features ({n_features}) excede a quantidade de eixos ' \
                      f'definidos por n_rows ({n_rows}) e n_cols ({n_cols})! \nSerão mostradas apenas {n_rows * n_cols} features.')
                return None
        target_idx = 0

        # Plotando, para cada eixo, um gráfico por classe target
        for target in targets:
            sns.distplot(target[feat], color=color_list[target_idx], hist=hist, ax=ax, label=target_names[target_idx])
            target_idx += 1

        # Incrementando índices
        j += 1
        if j == n_cols:
            j = 0
            i += 1

        # Customizando plotagem
        ax.set_title(f'Feature: {feat}', color='dimgrey', size=14)
        plt.setp(ax, yticks=[])
        sns.set(style='white')
        sns.despine(left=True)

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(n_rows * n_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= n_features:
            try:
                axs[i][j].axis('off')
            except TypeError as e:
                axs[j].axis('off')

        # Incrementando
        j += 1
        if j == n_cols:
            j = 0
            i += 1

    # Finalizando customização
    plt.tight_layout()
    plt.show()


# Função para análise da matriz de correlação
def correlation_matrix(data, label_name, top_k=10, fmt='.2f', cmap='YlGnBu', figsize=(18, 7),
                       cbar=True, annot=True, square=True):
    """
    Etapas:
        1. construção de correlação entre as variáveis
        2. filtragem das top k variáveis com maior correlação
        3. plotagem e configuração da matriz de correlação

    Argumentos:
        data -- DataFrame a ser analisado [pandas.DataFrame]
        label_name -- nome da coluna contendo a variável resposta [string]
        top_k -- indicador das top k variáveis a serem analisadas [int]
        fmt -- formato dos números de correlação na plotagem [string]
        cmap -- color mapping [string]
        figsize -- dimensões da plotagem gráfica [tupla]
        cbar -- indicador de plotagem da barra indicadora lateral [bool]
        annot -- indicador de anotação dos números de correlação na matriz [bool]
        square -- indicador para redimensionamento quadrático da matriz [bool]

    Retorno:
        None
    """

    # Criando matriz de correlação para a base de dados
    corr_mx = data.corr()

    # Retornando apenas as top k variáveis com maior correlação frente a variável resposta
    corr_cols = corr_mx.nlargest(top_k, label_name)[label_name].index
    corr_data = np.corrcoef(data[corr_cols].values.T)

    # Construindo plotagem da matriz
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, cbar=cbar, annot=annot, square=square, fmt=fmt, cmap=cmap,
                yticklabels=corr_cols.values, xticklabels=corr_cols.values)
    plt.show()