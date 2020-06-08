"""
    Script criado para consolidar funções úteis utilizadas na plotagem e personalização de gráficos
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
from typing import *
from dataclasses import dataclass
from math import ceil

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

# Classe para plotagem dos rótulos dos dados em gráficos de barras
# Referência: https://towardsdatascience.com/annotating-bar-charts-and-other-matplolib-techniques-cecb54315015
#Alias types to reduce typing, no pun intended
Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
Axis = matplotlib.axes.Axes
PosValFunc = Callable[[Patch], PosVal]

@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2
    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos
        ha = "center" if centered else  "left"
        self._annotate(ax, get_vals, ha=ha, va="center")
    def vertical(self, ax: Axis, centered:bool=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2,
                   p.get_y() + p.get_height() / div
            )
            return value, pos
        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)
    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color,
               "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            value, pos = func(p)
            ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


"""
--------------------------------------------
---------- 2. PLOTAGENS GRÁFICAS -----------
--------------------------------------------
"""


# Função para plotagem de gráfico de rosca em relação a uma variávei específica do dataset
def donut_plot(df, col, label_names, ax, text='', colors=['crimson', 'navy'], circle_radius=0.8):
    """
    Etapas:
        1. definição de funções úteis para mostrar rótulos em valor absoluto e porcentagem
        2. criação de figura e círculo central de raio pré-definido
        3. plotagem do gráfico de pizza e adição do círculo central
        4. configuração final da plotagem

    Argumentos:
        df -- DataFrame alvo da análise [pandas.DataFrame]
        col -- coluna do DataFrame a ser analisada [string]
        label_names -- nomes customizados a serem plotados como labels [list]
        text -- texto central a ser posicionado [string / default: '']
        colors -- cores das entradas [list / default: ['crimson', 'navy']]
        figsize -- dimensões da plotagem [tupla / default: (8, 8)]
        circle_radius -- raio do círculo central [float / default: 0.8]

    Retorno:
        None
    """

    # Definindo funções úteis para plotagem dos rótulos no gráfico
    def make_autopct(values):
        """
        Etapas:
            1. definição de função para formatação dos rótulos

        Argumentos:
            values -- valores extraídos da função value_counts() da coluna de análise [list]

        Retorno:
            my_autopct -- string formatada para plotagem dos rótulos
        """

        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))

            return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)

        return my_autopct

    # Retorno dos valores e definição da figura
    values = df[col].value_counts().values
    center_circle = plt.Circle((0, 0), circle_radius, color='white')

    # Plotando gráfico de rosca
    ax.pie(values, labels=label_names, colors=colors, autopct=make_autopct(values))
    ax.add_artist(center_circle)

    # Configurando argumentos do texto central
    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    ax.set_title(f'Gráfico de Rosca para {col}', size=14, color='dimgrey')


# Função para análise da matriz de correlação
def target_correlation_matrix(data, label_name, n_vars=10, corr='positive', fmt='.2f', cmap='YlGnBu', figsize=(18, 7),
                              cbar=True, annot=True, square=True):
    """
    Etapas:
        1. construção de correlação entre as variáveis
        2. filtragem das top k variáveis com maior correlação
        3. plotagem e configuração da matriz de correlação

    Argumentos:
        data -- DataFrame a ser analisado [pandas.DataFrame]
        label_name -- nome da coluna contendo a variável resposta [string]
        n_vars -- indicador das top k variáveis a serem analisadas [int]
        corr -- indicador booleano para plotagem de correlações ('positive', 'negative') [string]
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
    if corr == 'positive':
        corr_cols = list(corr_mx.nlargest(n_vars+1, label_name)[label_name].index)
        title = f'Top {n_vars} fatures com maior correlação positiva com a variável {label_name}'
    elif corr == 'negative':
        corr_cols = list(corr_mx.nsmallest(n_vars+1, label_name)[label_name].index)
        corr_cols = [label_name] + corr_cols[:-1]
        title = f'Top {n_vars} fatures com maior correlação negativa com a variável {label_name}'
        cmap = 'magma'

    corr_data = np.corrcoef(data[corr_cols].values.T)

    # Construindo plotagem da matriz
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, cbar=cbar, annot=annot, square=square, fmt=fmt, cmap=cmap,
                yticklabels=corr_cols, xticklabels=corr_cols)
    ax.set_title(title, size=14, color='dimgrey', pad=20)
    plt.show()

    return corr_cols[1:]


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


# Função para plotagem de stripplot
def stripplot(data, label, features, n_rows, n_cols, figsize=(16, 8), palette='viridis'):
    """
    Etapas:
        1. criação de figura de acordo com as especificações dos argumentos
        2. laço para plotagem de striplot por eixo
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
        sns.stripplot(x=data[label], y=data[feat], ax=ax, palette=palette)

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


# Função responsável por plotar volumetria de uma variável categórica (quebra por hue é opcional)
def countplot(df, feature, order=True, hue=False, label_names=None, palette='plasma', colors=['darkgray', 'navy'],
              figsize=(12, 5), loc_legend='lower left', width=0.75, sub_width=0.3, sub_size=12):
    """
    Etapas:
        1. customização da plotagem de acordo com a presença (ou não) do parâmetro hue
        2. definição das figuras e plotagem dos gráficos adequados
        3. customização da plotagem

    Argumentos:
        df -- DataFrame alvo da análise [pandas.DataFrame]
        feature -- coluna a ser analisada [string]
        order -- flag booleano pra indicar a ordenação da plotagem [bool - default: True]
        hue -- parâmetro de quebra de análise [string - default: False]
        label_names -- descrição dos labels a serem colocados na legenda [list - default: None]
        palette -- paleta de cores a ser utilizada no plot singular da variável [string - default: 'viridis']
        colors -- cores a serem utilizadas no plot quebrado por hue [list - default: ['darkgray', 'navy']]
        figsize -- dimensões da plotagem [tupla - default: (15, 5)]
        loc_legend -- posição da legenda em caso de plotagem por hue [string - default: 'best']
        width -- largura das barras em caso de plotagem por hue [float - default: 0.5]
        sub_width -- parâmetro de alinhamento dos rótulos em caso de plotagem por hue [float - default: 0.3]

    Retorno:
    """

    # Verificando plotagem por quebra de alguma variável categórica
    ncount = len(df)
    if hue != False:
        # Redifinindo dimensões e plotando gráfico solo + versus variável categórica
        figsize = (figsize[0], figsize[1] * 2)
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
        if order:
            sns.countplot(x=feature, data=df, palette=palette, ax=axs[0], order=df[feature].value_counts().index)
        else:
            sns.countplot(x=feature, data=df, palette=palette, ax=axs[0])

        # Plotando gráfico de análise por hue (stacked bar chart)
        feature_rate = pd.crosstab(df[feature], df[hue])
        percent_df = feature_rate.div(feature_rate.sum(1).astype(float), axis=0)
        if order:
            sort_cols = list(df[feature].value_counts().index)
            sorter_index = dict(zip(sort_cols, range(len(sort_cols))))
            percent_df['rank'] = percent_df.index.map(sorter_index)
            percent_df = percent_df.sort_values(by='rank')
            percent_df = percent_df.drop('rank', axis=1)
            percent_df.plot(kind='bar', stacked=True, ax=axs[1], color=colors, width=width)
        else:
            percent_df.plot(kind='bar', stacked=True, ax=axs[1], color=colors, width=width)
        # sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=axs[1], order=df[feature].value_counts().index)

        # Inserindo rótulo de percentual para gráfico singular
        for p in axs[0].patches:
            # Coletando parâmetros e inserindo no gráfico
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            axs[0].annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y), ha='center', va='bottom',
                            size=sub_size)

        # Inserindo rótulo de percentual para gráfico hue
        for p in axs[1].patches:
            # Coletando parâmetros
            height = p.get_height()
            width = p.get_width()
            x = p.get_x()
            y = p.get_y()

            # Formatando parâmetros coletados e inserindo no gráfico
            label_text = f'{round(100 * height, 1)}%'
            label_x = x + width - sub_width
            label_y = y + height / 2
            axs[1].text(label_x, label_y, label_text, ha='center', va='center', color='white', fontweight='bold',
                        size=sub_size)

        # Definindo títulos
        axs[0].set_title(f'Análise de Volumetria da Variável {feature}', size=14, color='dimgrey', pad=20)
        axs[0].set_ylabel('Volumetria')
        axs[1].set_title(f'Análise de Volumetria da Variável {feature} por {hue}', size=14, color='dimgrey', pad=20)
        axs[1].set_ylabel('Percentual')

        # Formatando eixo de cada uma das plotagens
        for ax in axs:
            format_spines(ax, right_border=False)

        # Definindo legenda para hue
        plt.legend(loc=loc_legend, title=f'{hue}', labels=label_names)

    else:
        # Plotagem única: sem quebra por variável hue
        fig, ax = plt.subplots(figsize=figsize)
        if order:
            sns.countplot(x=feature, data=df, palette=palette, ax=ax, order=df[feature].value_counts().index)
        else:
            sns.countplot(x=feature, data=df, palette=palette, ax=ax)

            # Formatando eixos
        ax.set_ylabel('Volumetria')
        format_spines(ax, right_border=False)

        # Inserindo rótulo de percentual
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y), ha='center', va='bottom')

        # Definindo título
        ax.set_title(f'Análise de Volumetria da Variável {feature}', size=14, color='dimgrey')

    # Configurações finais
    plt.tight_layout()
    plt.show()


# Função para plotagem de volumetria das variáveis categóricas do conjunto de dados
def catplot_analysis(df, fig_cols, palette='viridis'):
    """
    Etapas:
        1. retorno das variáveis categóricas do conjunto de dados
        2. parametrização de variáveis de plotagem
        3. aplicação de laços de repetição para plotagens / formatação

    Argumentos:
        df -- conjunto de dados a ser analisado [pandas.DataFrame]
        fig_cols -- quantidade de colunas da figura matplotlib [int]

    Retorno:
        None
    """

    # Criando um DataFrame de variáveis categóricas
    cat_features = [col for col, dtype in df.dtypes.items() if dtype == 'object']
    df_categorical = df.loc[:, cat_features]

    # Retornando parâmetros para organização da figura
    total_cols = df_categorical.shape[1]
    fig_cols = 3
    fig_rows = ceil(total_cols / fig_cols)
    ncount = len(df)

    # Criando figura de plotagem
    fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=(fig_cols * 5, fig_rows * 4))
    i, j = 0, 0

    # Laço de repetição para plotagem categórica
    for col in cat_features:
        # Indexando variáveis e plotando gráfico
        ax = axs[i, j]
        sns.countplot(y=col, data=df_categorical, palette=palette, ax=ax,
                      order=df_categorical[col].value_counts().index)

        # Customizando gráfico
        format_spines(ax, right_border=False)
        AnnotateBars(n_dec=0, color='dimgrey').horizontal(ax)

        # Incrementando índices de eixo
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(fig_rows * fig_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= len(cat_features):
            try:
                axs[i][j].axis('off')
            except TypeError as e:
                axs[j].axis('off')

        # Incrementando
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    plt.tight_layout()
    plt.show()