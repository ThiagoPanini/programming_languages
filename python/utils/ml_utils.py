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
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from utils.viz_utils import *
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, cross_val_predict, \
                                    learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, \
    accuracy_score, precision_score, recall_score, f1_score


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

    def __init__(self, model, set_prep, features):
        self.model = model
        self.X_train = set_prep['X_train_prep']
        self.y_train = set_prep['y_train']
        self.X_test = set_prep['X_test_prep']
        self.y_test = set_prep['y_test']
        self.features = features
        self.model_name = model.__class__.__name__

    def random_search(self, scoring, param_grid=None, cv=5):
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
        """if tree:
            param_grid = {
                'criterion': ['entropy', 'gini'],
                'max_depth': [3, 4, 5, 8, 10],
                'max_features': np.arange(1, self.X_train.shape[1]),
                'class_weight': ['balanced', None]
            }"""

        # Aplicando busca aleatória dos hiperparâmetros
        rnd_search = RandomizedSearchCV(self.model, param_grid, scoring=scoring, cv=cv, verbose=1,
                                        random_state=42, n_jobs=-1)
        rnd_search.fit(self.X_train, self.y_train)

        return rnd_search.best_estimator_

    def fit(self, rnd_search=False, scoring=None, param_grid=None):
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
            print(f'Treinando modelo {self.model_name} com RandomSearchCV.')
            self.trained_model = self.random_search(param_grid=param_grid, scoring=scoring)
            print(f'Treinamento finalizado com sucesso! Configurações do modelo: \n\n{self.trained_model}')
        else:
            print(f'Treinando modelo {self.model_name}.')
            self.trained_model = self.model.fit(self.X_train, self.y_train)
            print(f'Treinamento finalizado com sucesso! Configurações do modelo: \n\n{self.trained_model}')

    def evaluate_performance(self, cv=5, test=False):
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

        if test:
            # Retornando predições com os dados de teste
            y_pred = self.trained_model.predict(self.X_test)

            # Retornando métricas para os dados de teste
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            approach = 'Test Set'
        else:
            # Avaliando principais métricas do modelo através de validação cruzada
            accuracy = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                       scoring='accuracy').mean()
            precision = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                        scoring='precision').mean()
            recall = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                     scoring='recall').mean()
            f1 = cross_val_score(self.trained_model, self.X_train, self.y_train, cv=cv,
                                 scoring='f1').mean()
            approach = f'Train (CV={cv})'

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

        # Salvando dados em um DataFrame
        performance = {}
        performance['approach'] = approach
        performance['acc'] = round(accuracy, 4)
        performance['precision'] = round(precision, 4)
        performance['recall'] = round(recall, 4)
        performance['f1'] = round(f1, 4)
        performance['auc'] = round(auc, 4)
        performance['total_time'] = round(delta_time, 3)

        df_performance = pd.DataFrame(performance, index=performance.keys()).reset_index(drop=True).loc[:0, :]
        df_performance.index = [self.model_name]

        return df_performance

    def plot_confusion_matrix(self, classes, cv=5, cmap=plt.cm.Blues, title='Confusion Matrix', normalize=False):
        """
        Etapas:
            1. cálculo de matriz de confusão utilizando predições com cross-validation
            2. configuração e construção de plotagem
            3. formatação dos labels da plotagem

        Argumentos:
            classes -- nome das classes envolvidas no modelo [list]
            cv -- número de folds aplicados na validação cruzada [int - default: 5]
            cmap -- mapeamento colorimétrico da matriz [plt.colormap - default: plt.cm.Blues]
            title -- título da matriz de confusão [string - default: 'Confusion Matrix']
            normaliza -- indicador para normalização dos dados da matriz [bool - default: False]

        Retorno
        """

        # Realizando predições e retornando matriz de confusão
        y_pred = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv)
        conf_mx = confusion_matrix(self.y_train, y_pred)

        # Plotando matriz
        sns.set(style='white', palette='muted', color_codes=True)
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizando eixos
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title, size=14)

    def plot_roc_curve(self, cv=5):
        """
        Etapas:
            1. retorno dos scores do modelo utilizando predição por validação cruzada
            2. encontro das taxas de falsos positivos e verdadeiros negativos
            3. cálculo da métrica AUC e plotagem da curva ROC

        Argumentos:
            cv -- número de k-folds utilizados na validação cruzada [int - default: 5]

        Retorno:
            None
        """

        # Calculando scores utilizando predição por validação cruzada
        try:
            y_scores = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv,
                                         method='decision_function')
        except:
            # Algoritmos baseados em Árvore não possuem o methodo "decision_function"
            y_probas = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv=cv,
                                         method='predict_proba')
            y_scores = y_probas[:, 1]

        # Calculando taxas de falsos positivos e verdadeiros positivos
        fpr, tpr, thresholds = roc_curve(self.y_train, y_scores)
        auc = roc_auc_score(self.y_train, y_scores)

        # Plotando curva ROC
        plt.plot(fpr, tpr, linewidth=2, label=f'{self.model_name} auc={auc: .3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.02, 1.02, -0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

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

    def plot_learning_curve(self, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10),
                            figsize=(12, 6)):
        """
        Etapas:
            1. cálculo dos scores de treino e validação de acordo com a quantidade m de dados
            2. cálculo de parâmetros estatísticos (média e desvio padrão) dos scores
            3. plotagem da curva de aprendizado de treino e validação

        Argumentos:
            y_lim -- definição de limites do eixo y [list - default: None]
            cv -- k folds na aplicação de validação cruzada [int - default: 5]
            n_jobs -- número de jobs durante a execução da função learning_curve [int - default: 1]
            train_sizes -- tamanhos considerados para as fatias do dataset [np.array - default: linspace(.1, 1, 10)]
            figsize -- dimensões da plotagem gráfica [tupla - default: (12, 6)]

        Retorno:
            None
        """

        # Retornando parâmetros de scores de treino e validação
        train_sizes, train_scores, val_scores = learning_curve(self.trained_model, self.X_train, self.y_train,
                                                               cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        # Calculando médias e desvios padrão (treino e validação)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Plotando gráfico de curva de aprendizado
        fig, ax = plt.subplots(figsize=figsize)

        # Resultado em dados de treino
        ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
        ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                        alpha=0.1, color='blue')

        # Resultado em validação cruzada
        ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
        ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                        alpha=0.1, color='crimson')

        # Customizando gráfico
        ax.set_title(f'Modelo {self.model_name} - Curva de Aprendizado', size=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc='best')
        plt.show()