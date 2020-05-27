"""
    Script criado para consolidar funções úteis utilizadas no treinamento dos mais variados modelos de
    Machine Learning.
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.viz_utils import *
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report


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
    

"""
--------------------------------------------
------- 1. MODELOS DE CLASSIFICAÇÃO --------
--------------------------------------------
"""


class BinaryBaselineClassifier():

    def __init__(self, baseline_model, X, y, features):
        self.baseline_model = baseline_model
        self.X = X
        self.y = y
        self.features = features
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.20,
                                                                                random_state=42)

    def random_search(self, scoring, param_grid=None, tree=True, cv=5):
        """
        Etapas:
            1. definição automática de parâmetros de busca caso o modelo sejá uma Árvore de Decisão
            2. aplicação de RandomizedSearchCV com os parâmetros definidos

        Argumentos:
            scoring -- métrica a ser otimizada durante a busca [string]
            param_grid -- dicionário com os parâmetros a serem utilizados na busca [dict]
            tree -- flag para indicar se o modelo baseline é uma árvore de decisão [bool]

        Retorno:
            best_estimator_ -- melhor modelo encontrado na busca
        """

        # Validando baseline como Árvore de Decisão (grid definido automaticamente)
        if tree:
            param_grid = {
                'criterion': ['entropy', 'gini'],
                'max_depth': [3, 4, 5, 8, 10],
                'max_features': np.arange(1, X_train.shape[1])
            }

        # Aplicando busca aleatória dos hiperparâmetros
        rnd_search = RandomizedSearchCV(self.baseline_model, param_grid, scoring=scoring, cv=cv, random_state=42)
        rnd_search.fit(self.X_train, self.y_train)

        return rnd_search.best_estimator_

    def fit(self, rnd_search=False, scoring=None, param_grid=None, tree=True):
        """
        Etapas:
            1. treinamento do modelo e atribuição do resultado como um atributo da classe

        Argumentos:
            rnd_search -- flag indicativo de aplicação de RandomizedSearchCV [bool]
            scoring -- métrica a ser otimizada durante a busca [string]
            param_grid -- dicionário com os parâmetros a serem utilizados na busca [dict]
            tree -- flag para indicar se o modelo baseline é uma árvore de decisão [bool]

        Retorno:
            None
        """

        # Treinando modelo de acordo com o argumento selecionado
        if rnd_search:
            self.trained_model = self.random_search(scoring=scoring)
        else:
            self.trained_model = self.baseline_model.fit(self.X_train, self.y_train)

    def evaluate_performance(self, cv=5):
        """
        Etapas:
            1. medição das principais métricas pro modelo

        Argumentos:
            cv -- número de k-folds durante a aplicação do cross validation [int]

        Retorno:
            df_performance -- DataFrame contendo a performance do modelo frente as métricas [pandas.DataFrame]
        """

        # Iniciando medição de tempo
        t0 = time.time()

        # Avaliando principais métricas do modelo através de validação cruzada
        accuracy = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                   scoring='accuracy').mean()
        precision = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                    scoring='precision').mean()
        recall = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                 scoring='recall').mean()
        f1 = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                             scoring='f1').mean()

        # AUC score
        try:
            y_scores = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv,
                                         method='decision_function')
        except:
            # Modelos baseados em árvores não possuem o método 'decision_function', mas sim 'predict_proba'
            y_probas = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv,
                                         method='predict_proba')
            y_scores = y_probas[:, 1]
        # Calculando AUC
        auc = roc_auc_score(self.y_train, y_scores)

        # Finalizando medição de tempo
        t1 = time.time()
        delta_time = t1 - t0
        model_name = self.trained_model.__class__.__name__

        # Salvando dados em um DataFrame
        performance = {}
        performance['acc'] = round(accuracy, 4)
        performance['precision'] = round(precision, 4)
        performance['recall'] = round(recall, 4)
        performance['f1'] = round(f1, 4)
        performance['auc'] = round(auc, 4)
        performance['total_time'] = round(delta_time, 3)

        df_performance = pd.DataFrame(performance, index=performance.keys()).reset_index(drop=True).loc[:0, :]
        df_performance.index = [model_name]

        return df_performance

    def confusion_matrix(self, labels, cv=5, cmap=plt.cm.Blues, normalize=False, figsize=(6, 5)):
        """
        Etapas:

        Argumentos:

        Retorno
        """

        # Realizando predições e retornando matriz de confusão
        y_pred = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv)
        conf_mx = confusion_matrix(self.y_train, y_pred)

        # Plotando matriz
        plt.figure(figsize=figsize)
        sns.set(style='white', palette='muted', color_codes=True)
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(labels))

        # Customizando eixos
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(self.trained_model.__class__.__name__ + '\nConfusion Matrix', size=14)
        plt.show()

        return conf_mx

    def feature_importance_analysis(self):
        """
        Etapas:
            1. retorno de importância das features
            2. construção de um DataFrame com as features mais importantes pro modelo

        Argumentos:
            None

        Retorno:
            feat_imp -- DataFrame com feature importances [pandas.DataFrame]
        """

        # Retornando feature importance do modelo
        importances = self.trained_model.feature_importances_
        feat_imp = pd.DataFrame({})
        feat_imp['feature'] = self.features
        feat_imp['importance'] = importances
        feat_imp = feat_imp.sort_values(by='importance', ascending=False)
        feat_imp.reset_index(drop=True, inplace=True)

        return feat_imp