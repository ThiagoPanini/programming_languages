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
from datetime import datetime
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

    def evaluate_performance(self, approach, cv=5, test=False):
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
            y_proba = self.trained_model.predict_proba(self.X_test)[:, 1]

            # Retornando métricas para os dados de teste
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
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


"""
--------------------------------------------
------ 2. ANÁLISE DE CLASSIFICADORES -------
--------------------------------------------
"""

class ClassifiersAnalysis():

    def __init__(self, set_classifiers, set_prep, features):
        self.set_classifiers = set_classifiers
        self.X_train = set_prep['X_train_prep']
        self.y_train = set_prep['y_train']
        self.X_test = set_prep['X_test_prep']
        self.y_test = set_prep['y_test']
        self.features = features
        self.trained_classifiers = {}
        self.classifiers_performance = {}
        for model_name, model_info in self.set_classifiers.items():
            self.trained_classifiers[model_name] = {}

    def fit(self, rnd_search=False, scoring='roc_auc', cv=5, n_jobs=-1, verbose=5, random_state=42):
        """
        Etapas:
            1. treinamento de cada um dos modelos armazenados no set de classificadores

        Argumentos:
            rnd_search -- flag indicativo de aplicação de RandomizedSearchCV [bool, default: False]
            scoring -- métrica a ser otimizada na busca dos hyperparâmetros [string, default: 'roc_auc']
            cv -- número de k-folds a ser utilizado na validação cruzada [int, default: 5]
            n_jobs -- número de workers no processamento do grid [int, default: -1]
            verbose -- indicador de comunicação durante a busca [int, default: 5]
            random_state -- semente aleatória [int, default: 42]

        Retorno:
            None
        """

        # Comunicação inicial de treinamento de modelos
        print('------------ Treinamento de Classificadores Selecionados ------------\n')
        i = 1

        # Iterando sobre cada um dos modelos selecionados
        for model_name, model_info in self.set_classifiers.items():

            # Parâmetros para controle de verbosity
            t0 = datetime.now()
            t0_fmt = t0.strftime("%Hh:%Mm:%Ss")
            print(f'{i}. {model_name}')
            print(f'   RandomSearhCV: {rnd_search} - Início: {t0_fmt}')

            # Validando aplicação de Random Search
            if rnd_search:
                # Início da busca pelo melhor conjunto de hyperparâmetros
                random_search = RandomizedSearchCV(model_info['model'], model_info['params'], scoring=scoring, cv=cv,
                                                   verbose=verbose, random_state=random_state, n_jobs=n_jobs)
                random_search.fit(self.X_train, self.y_train)

                # Salvando melhor modelo em atributo da classe
                self.trained_classifiers[model_name]['estimator'] = random_search.best_estimator_

            else:
                # Treinamento sem busca por Random Search
                self.trained_classifiers[model_name]['estimator'] = model_info['model'].fit(self.X_train, self.y_train)

            # Parâmetros para controle de verbosity
            t1 = datetime.now()
            t1_fmt = t1.strftime("%Hh:%Mm:%Ss")
            print(f'   Treinamento finalizado - Fim: {t1_fmt}. Tempo total: {(t1 - t0)}\n')
            i += 1

    def compute_cv_performance(self, model_name, trained_model, cv):
        """
        Etapas:
            1. cálculo das principais métricas utilizando validação ruzada
            2. salvando resultados em um objeto do tipo DataFrame

        Argumentos:
            model_name -- nome do classificador avaliado [string]
            trained_model -- classificador já treinado [sklearn.Classifier]

        Retorno:
            cv_performance -- métricas consolidadas [pandas.DataFrame]
        """

        # Calculando principais métricas utilizando validação cruzada
        t0 = time.time()
        accuracy = cross_val_score(trained_model, self.X_train, self.y_train, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(trained_model, self.X_train, self.y_train, cv=cv, scoring='precision').mean()
        recall = cross_val_score(trained_model, self.X_train, self.y_train, cv=cv, scoring='recall').mean()
        f1 = cross_val_score(trained_model, self.X_train, self.y_train, cv=cv, scoring='f1').mean()
        try:
            y_scores = cross_val_predict(trained_model, self.X_train, self.y_train, cv=cv, method='decision_function')
        except:
            # Modelos baseados em árvores não possuem o método 'decision_function', mas sim 'predict_proba'
            y_probas = cross_val_predict(trained_model, self.X_train, self.y_train, cv=cv, method='predict_proba')
            y_scores = y_probas[:, 1]
        auc = roc_auc_score(self.y_train, y_scores)

        # Salvando probabilidades nos stats dos classificadores treinados
        self.trained_classifiers[model_name]['train_scores'] = y_scores

        # Criação de DataFrame para alocação das métricas
        t1 = time.time()
        delta_time = t1 - t0
        train_performance = {}
        train_performance['model'] = model_name
        train_performance['approach'] = f'Treino cv={cv}'
        train_performance['acc'] = round(accuracy, 4)
        train_performance['precision'] = round(precision, 4)
        train_performance['recall'] = round(recall, 4)
        train_performance['f1'] = round(f1, 4)
        train_performance['auc'] = round(auc, 4)
        train_performance['total_time'] = round(delta_time, 3)

        # Salvando métricas no respectivo dicionário do classificador

        return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

    def compute_test_performance(self, model_name, trained_model):
        """
        Etapas:
            1. cálculo das principais métricas utilizando os dados de teste
            2. salvando resultados em um objeto do tipo DataFrame

        Argumentos:
            model_name -- nome do classificador avaliado [string]
            trained_model -- classificador já treinado [sklearn.Classifier]

        Retorno:
            cv_performance -- métricas consolidadas [pandas.DataFrame]
        """

        # Retornando predições com os dados de teste
        t0 = time.time()
        y_pred = trained_model.predict(self.X_test)
        y_proba = trained_model.predict_proba(self.X_test)
        y_scores = y_proba[:, 1]

        # Retornando métricas para os dados de teste
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_scores)

        # Salvando probabilidades nos stats dos classificadores treinados
        self.trained_classifiers[model_name]['test_scores'] = y_scores

        # Criação de DataFrame para alocação das métricas
        t1 = time.time()
        delta_time = t1 - t0
        test_performance = {}
        test_performance['model'] = model_name
        test_performance['approach'] = f'Teste'
        test_performance['acc'] = round(accuracy, 4)
        test_performance['precision'] = round(precision, 4)
        test_performance['recall'] = round(recall, 4)
        test_performance['f1'] = round(f1, 4)
        test_performance['auc'] = round(auc, 4)
        test_performance['total_time'] = round(delta_time, 3)

        return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

    def evaluate_performance(self, cv=5):
        """
        Etapas:
            1. iteração sobre os classificadores treinados em dicionário da classe
            2. retorno das métricas utilizando validação cruzada
            3. retorno das métricas utilizando os dados de teste

        Argumentos:
            cv -- k-folds utilizados na validação cruzada [int, default: 5]

        Retorno:
            df_performances -- resultado das métricas dos modelos em treino e teste [pandas.DataFrame]
        """

        # Comunicação inicial de treinamento de modelos
        print('------------ Avaliação de Classificadores Selecionados ------------\n')
        i = 1
        df_performances = pd.DataFrame({})

        # Iterando sobre cada um dos classificadores do set
        for model_name, model_stats in self.trained_classifiers.items():
            # Início da avaliação
            t0 = time.time()
            t0_fmt = datetime.now().strftime("%Hh:%Mm:%Ss")
            print(f'{i}. {model_name}')
            print(f'   Início: {t0_fmt}\n')

            # Retornando performance via cross validation
            cv_performance = self.compute_cv_performance(model_name, trained_model=model_stats['estimator'], cv=cv)
            self.trained_classifiers[model_name]['train_performance'] = cv_performance

            # Retornando performance nos dados de teste
            test_performance = self.compute_test_performance(model_name, trained_model=model_stats['estimator'])
            self.trained_classifiers[model_name]['test_performance'] = test_performance

            # Unindo DataFrames
            model_performance = cv_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)
            i += 1

        return df_performances

    def plot_roc_curve(self, cv=5):
        """
        Etapas:
            1. retorno dos scores do modelo utilizando predição por validação cruzada
            2. encontro das taxas de falsos positivos e verdadeiros negativos
            3. cálculo da métrica AUC e plotagem da curva ROC

        Argumentos:
            cv -- número de k-folds utilizados na validação cruzada [int, default: 5]

        Retorno:
            None
        """

        # Iterando por cada classificador do set
        for model_name, model_stats in self.trained_classifiers.items():
            # Retornando scores
            y_scores = self.trained_classifiers[model_name]['train_scores']

            # Calculando taxas de falsos positivos e verdadeiros positivos
            fpr, tpr, thresholds = roc_curve(self.y_train, y_scores)
            auc = roc_auc_score(self.y_train, y_scores)

            # Plotando curva ROC
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} auc={auc: .3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Dados de Treino com Validação Cruzada')
            plt.annotate('Área sob a curva (AUC) de 50%\n(Score de um modelo aleatório)',
                         xy=(0.5, 0.5), xytext=(0.6, 0.4), arrowprops=dict(facecolor='#6E726D', shrink=0.05))
            plt.legend()

    def plot_score_distribution(self, model_name):
        """
        Etapas:
            1. retorno dos scores dos modelos em dicionário de classificadores treinados
            2. plotagem das distribuições de scores de treino e teste pra cada modelo

        Argumentos:
            model_name -- chave do dicionário do modelo treinado [string]

        Retorno:
            None
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.trained_classifiers[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.trained_classifiers.keys())}')
            return None

        # Retornando scores de treino e de teste
        train_scores = self.trained_classifiers[model_name]['train_scores']
        test_scores = self.trained_classifiers[model_name]['test_scores']

        # Plotando distribuição de scores
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        sns.kdeplot(train_scores[self.y_train == 1], ax=axs[0], label='y=1', color='darkslateblue')
        sns.kdeplot(train_scores[self.y_train == 0], ax=axs[0], label='y=0', color='crimson')
        sns.kdeplot(test_scores[self.y_test == 1], ax=axs[1], label='y=1', color='darkslateblue')
        sns.kdeplot(test_scores[self.y_test == 0], ax=axs[1], label='y=0', color='crimson')

        # Customizando plotagem
        format_spines(axs[0])
        format_spines(axs[1])
        axs[0].set_title('Distribuição de Scores - Dados de Treino', size=12, color='dimgrey')
        axs[1].set_title('Distribuição de Scores - Dados de Teste', size=12, color='dimgrey')
        plt.suptitle(f'Análise de Scores - Modelo {model_name}\n', size=14, color='black')
        plt.show()

    def save_class_distrib(self):
        """
        Etapas:

        Argumentos:

        Retorno:
        """

        # Retornando análise estatística sobre o score de cada classe
        for model_name, model_stats in self.trained_classifiers.items():
            # Retornando scores da classe positiva - Dados de treino
            train_scores_neg_class = model_stats['train_scores'][self.y_train == 0]
            train_scores_pos_class = model_stats['train_scores'][self.y_train == 1]

            # Criando DataFrame para consolidar estatísticas
            train_distrib_scores = pd.DataFrame({})
            train_distrib_scores['negativa'] = pd.Series(train_scores_neg_class).describe()
            train_distrib_scores['positiva'] = pd.Series(train_scores_pos_class).describe()
            trained_classifiers[model_name]['train_distrib_scores'] = train_distrib_scores

            # Retornando scores da classe positiva - Dados de teste
            test_scores_neg_class = model_stats['test_scores'][self.y_test == 0]
            test_scores_pos_class = model_stats['test_scores'][self.y_test == 1]

            # Criando DataFrame para consolidar estatísticas
            test_distrib_scores = pd.DataFrame({})
            test_distrib_scores['negativa'] = pd.Series(test_scores_neg_class).describe()
            test_distrib_scores['positiva'] = pd.Series(test_scores_pos_class).describe()
            trained_classifiers[model_name]['test_distrib_scores'] = train_distrib_scores

    def plot_score_bins(self, model_name, bin_range):
        """
        Etapas:

        Argumentos:

        Retorno:
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.trained_classifiers[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.trained_classifiers.keys())}')
            return None

        # Criando array de bins
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins))
                       if i > 0]

        # Retornando scores de treino e criando um DataFrame
        train_scores = self.trained_classifiers[model_name]['train_scores']
        df_train_scores = pd.DataFrame({})
        df_train_scores['scores'] = train_scores
        df_train_scores['target'] = self.y_train
        df_train_scores['faixa'] = pd.cut(train_scores, bins, labels=bins_labels)

        # Calculando distribuição por cada faixa - treino
        df_train_rate = pd.crosstab(df_train_scores['faixa'], df_train_scores['target'])
        df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis=0)

        # Retornando scores de teste e criando um DataFrame
        test_scores = self.trained_classifiers[model_name]['test_scores']
        df_test_scores = pd.DataFrame({})
        df_test_scores['scores'] = test_scores
        df_test_scores['target'] = self.y_test
        df_test_scores['faixa'] = pd.cut(test_scores, bins, labels=bins_labels)

        # Calculando distribuição por cada faixa - teste
        df_test_rate = pd.crosstab(df_test_scores['faixa'], df_test_scores['target'])
        df_test_percent = df_test_rate.div(df_test_rate.sum(1).astype(float), axis=0)

        # Definindo figura de plotagem
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Plotando gráficos de volumetria de cada classe por faixa
        for df_scores, ax in zip([df_train_scores, df_test_scores], [axs[0, 0], axs[1, 0]]):
            sns.countplot(x='faixa', data=df_scores, hue='target', ax=ax, palette=['darkslateblue', 'crimson'])
            AnnotateBars(n_dec=0, color='dimgrey').vertical(ax)
            ax.legend(loc='upper right')
            ax.set_title('Volumetria das Classes por Faixa', size=12, color='dimgrey')
            format_spines(ax, right_border=False)

        # Plotando percentual de representatividade de cada classe por faixa
        for df_percent, ax in zip([df_train_percent, df_test_percent], [axs[0, 1], axs[1, 1]]):
            df_percent.plot(kind='bar', ax=ax, stacked=True, color=['darkslateblue', 'crimson'], width=0.6)

            # Customizando plotagem
            for p in ax.patches:
                # Coletando parâmetros para rótulos
                height = p.get_height()
                width = p.get_width()
                x = p.get_x()
                y = p.get_y()

                # Formatando parâmetros coletados e inserindo no gráfico
                label_text = f'{round(100 * height, 1)}%'
                label_x = x + width - 0.30
                label_y = y + height / 2
                ax.text(label_x, label_y, label_text, ha='center', va='center', color='white',
                        fontweight='bold', size=10)
            format_spines(ax, right_border=False)

        # Definições finais
        axs[0, 0].set_title('Volumetria das Classes por Faixa - Treino', size=12, color='dimgrey')
        axs[1, 0].set_title('Volumetria das Classes por Faixa - teste', size=12, color='dimgrey')
        axs[0, 1].set_title('Percentual das Classes por Faixa - Treino', size=12, color='dimgrey')
        axs[1, 1].set_title('Percentual das Classes por Faixa - Teste', size=12, color='dimgrey')
        # plt.suptitle(f'Análise Detalhada de Scores - {model_name}', size=14, color='black')
        plt.tight_layout()
        plt.show()