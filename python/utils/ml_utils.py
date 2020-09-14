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
import shap

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
------ 2. ANÁLISE DE CLASSIFICADORES -------
--------------------------------------------
"""


class BinaryClassifiersAnalysis():

    def __init__(self):
        self.classifiers_info = {}

    def fit(self, classifiers, X, y, approach='', random_search=False, scoring='roc_auc', cv=5, verbose=5, n_jobs=-1):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Iterando sobre cada modelo no dicionário de classificadores
        for model_name, model_info in classifiers.items():
            clf_key = model_name + approach
            print(f'Training model {clf_key}\n')
            self.classifiers_info[clf_key] = {}

            # Validando aplicação de RandomizedSearchCV
            if random_search:
                rnd_search = RandomizedSearchCV(model_info['model'], model_info['params'], scoring=scoring, cv=cv,
                                                verbose=verbose, random_state=42, n_jobs=n_jobs)
                rnd_search.fit(X, y)
                self.classifiers_info[clf_key]['estimator'] = rnd_search.best_estimator_
            else:
                self.classifiers_info[clf_key]['estimator'] = model_info['model'].fit(X, y)

    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Computando as principais métricas por validação cruzada
        t0 = time.time()
        accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
        precision = cross_val_score(estimator, X, y, cv=cv, scoring='precision').mean()
        recall = cross_val_score(estimator, X, y, cv=cv, scoring='recall').mean()
        f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1').mean()

        # Probabilidades para o cálculo da AUC
        try:
            y_scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
        except:
            # Modelos baseados em árvores não possuem o método 'decision_function', mas sim 'predict_proba'
            y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            y_scores = y_probas[:, 1]
        auc = roc_auc_score(y, y_scores)

        # Salvando scores no dicionário do classificador
        self.classifiers_info[model_name]['train_scores'] = y_scores

        # Criando DataFrame com as métricas
        t1 = time.time()
        delta_time = t1 - t0
        train_performance = {}
        train_performance['model'] = model_name
        train_performance['approach'] = f'Treino {cv} K-folds'
        train_performance['acc'] = round(accuracy, 4)
        train_performance['precision'] = round(precision, 4)
        train_performance['recall'] = round(recall, 4)
        train_performance['f1'] = round(f1, 4)
        train_performance['auc'] = round(auc, 4)
        train_performance['total_time'] = round(delta_time, 3)

        df_train_performance = pd.DataFrame(train_performance,
                                            index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

        return df_train_performance

    def compute_test_performance(self, model_name, estimator, X, y, cv=5):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Calculando predições e scores com dados de treino
        t0 = time.time()
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)
        y_scores = y_proba[:, 1]

        # Retornando métricas para os dados de teste
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_scores)

        # Salvando probabilidades nos stats dos classificadores treinados
        self.classifiers_info[model_name]['test_scores'] = y_scores

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

        df_test_performance = pd.DataFrame(test_performance,
                                           index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

        return df_test_performance

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5, approach='',
                             save=False, overwrite=False, performances_filepath='model_performances.csv'):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Iterando sobre cada classificador já treinado
        df_performances = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():

            # Verificando se o modelo já foi avaliado anteriormente
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            # Indexando variáveis para os cálculos
            print(f'Evaluating model {model_name}\n')
            estimator = model_info['estimator']

            # Retornando métricas nos dados de treino
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train)
            test_performance = self.compute_test_performance(model_name, estimator, X_test, y_test)

            # Salvando resultados no dicionário do modelo
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['test_performance'] = test_performance

            # Retornando DataFrame único com as performances obtidas
            model_performance = train_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)

            # Salvando conjuntos de dados como atributos para acesso futuro
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            model_info['model_data'] = model_data

        # Salvando DataFrame de performances se aplicável
        if save:
            # Adicionando informações de data da medição
            cols_performance = list(df_performances.columns)
            df_performances['anomesdia'] = datetime.now().strftime('%Y%m%d')
            df_performances['anomesdia_datetime'] = datetime.now()
            df_performances = df_performances.loc[:, ['anomesdia', 'anomesdia_datetime'] + cols_performance]

            # Validando sobrescrita ou append do log de performances já salvo
            if overwrite:
                df_performances.to_csv(performances_filepath, index=False)
            else:
                # Lendo base já existente e aplicando append
                try:
                    log_performances = pd.read_csv(performances_filepath)
                    full_performances = log_performances.append(df_performances)
                    full_performances.to_csv(performances_filepath, index=False)
                except FileNotFoundError:
                    print('Log de performances do modelo não existente no caminho especificado. Salvando apenas o atual.')
                    df_performances.to_csv(performances_filepath, index=False)

        return df_performances

    def feature_importance_analysis(self, features, specific_model=None, graph=True, ax=None, top_n=30,
                                    palette='viridis', save=False, features_filepath='features_info.csv'):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Iterando sobre cada um dos classificadores já treinados
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():
            # Criando DataFrame com as features importances
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                continue
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['anomesdia'] = datetime.now().strftime('%Y%m')
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp.reset_index(drop=True, inplace=True)

            # Salvando set de feature importances no dicionário do classificador
            self.classifiers_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            all_feat_imp['model'] = model_name

        # Retornando feature importances de um classificador específico
        if specific_model is not None:
            try:
                model_feature_importance = self.classifiers_info[specific_model]['feature_importances']
                if graph:  # Plotando gráfico
                    sns.barplot(x='importance', y='feature', data=model_feature_importance.iloc[:top_n, :],
                                ax=ax, palette=palette)
                    format_spines(ax, right_border=False)
                    ax.set_title(f'Top {top_n} {specific_model} Features mais Relevantes', size=14, color='dimgrey')

                # Saving features for a specific model
                if save:
                    model_feature_importance['model'] = specific_model
                    order_cols = ['anomesdia', 'anomesdia_datetime', 'model', 'feature', 'importance']
                    model_feature_importance = model_feature_importance.loc[:, order_cols]
                    model_feature_importance.to_csv(features_filepath, index=False)

                return model_feature_importance

            # Exceção caso a chave passada no parâmetro "specific_model" não esteja presente no dicionário
            except:
                print(f'Classificador {specific_model} não existente nas chaves de classificadores treinados.')
                print(f'Opções possíveis: {list(self.classifiers_info.keys())}')
                return None

        # Validando o salvamento das feature importances de todos os modelos caso "specific_model" não esteja definido
        else:
            if save:
                order_cols = ['anomesdia', 'anomedia_datetime', 'model', 'feature', 'importance']
                all_feat_imp = all_feat_imp.loc[:, order_cols]
                all_feat_imp.to_csv(features_filepath, index=False)

        # Validando combinação incoerente de argumentos
        if graph and specific_model is None:
            print('Por favor, escolha um modelo específico para visualizar o gráfico das feature importances')
            return None

    def plot_roc_curve(self, figsize=(16, 6)):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Criando figura para plotagem da curva ROC
        fig, axs = plt.subplots(ncols=2, figsize=figsize)

        # Iterando sobre cada um dos classificadores treinados
        for model_name, model_info in self.classifiers_info.items():
            # Retornando conjuntos y do modelo
            y_train = model_info['model_data']['y_train']
            y_test = model_info['model_data']['y_test']

            # Retornando scores
            train_scores = model_info['train_scores']
            test_scores = model_info['test_scores']

            # Calculando taxas de falsos positivos e verdadeiros positivos
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
            test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_scores)

            # Retornando AUC pra treino e teste
            train_auc = model_info['train_performance']['auc'].values[0]
            test_auc = model_info['test_performance']['auc'].values[0]

            # Plotando gráfico (dados de treino)
            plt.subplot(1, 2, 1)
            plt.plot(train_fpr, train_tpr, linewidth=2, label=f'{model_name} auc={train_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Train Data')
            plt.legend()

            # Plotando gráfico (dados de teste)
            plt.subplot(1, 2, 2)
            plt.plot(test_fpr, test_tpr, linewidth=2, label=f'{model_name} auc={test_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Test Data', size=12)
            plt.legend()

        plt.show()

    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Retornando matriz de confusão
        conf_mx = confusion_matrix(y_true, y_pred)

        # Plotando matriz
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
        plt.title(f'{model_name}\nConfusion Matrix', size=12)

    def plot_confusion_matrix(self, classes, normalize=False, cmap=plt.cm.Blues):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(10, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterando por cada um dos classificadores
        for model_name, model_info in self.classifiers_info.items():
            # Retornando dados em cada modelo
            X_train = model_info['model_data']['X_train']
            y_train = model_info['model_data']['y_train']
            X_test = model_info['model_data']['X_test']
            y_test = model_info['model_data']['y_test']

            # Realizando predições e retornando matriz de confusão
            train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
            test_pred = model_info['estimator'].predict(X_test)

            # Plotando matriz (dados de treino)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, cmap=cmap,
                                         normalize=normalize)
            k += 1

            # Plotando matriz (dados de teste)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Test', y_test, test_pred, classes=classes, cmap=plt.cm.Greens,
                                         normalize=normalize)
            k += 1

        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, model_name, ax, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.classifiers_info.keys())}')
            return None

        # Retornando dados em cada modelo
        X_train = model['model_data']['X_train']
        y_train = model['model_data']['y_train']
        X_test = model['model_data']['X_test']
        y_test = model['model_data']['y_test']

        # Retornando parâmetros de scores de treino e validação
        train_sizes, train_scores, val_scores = learning_curve(model['estimator'], X_train, y_train, cv=cv,
                                                               n_jobs=n_jobs, train_sizes=train_sizes)

        # Calculando médias e desvios padrão (treino e validação)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Resultado em dados de treino
        ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
        ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                        alpha=0.1, color='blue')

        # Resultado em validação cruzada
        ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
        ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                        alpha=0.1, color='crimson')

        # Customizando gráfico
        ax.set_title(f'Model {model_name} - Learning Curve', size=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc='best')

    def plot_score_distribution(self, model_name, shade=False):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.classifiers_info.keys())}')
            return None

        # Retornando conjuntos y do modelo
        y_train = self.classifiers_info[model_name]['model_data']['y_train']
        y_test = self.classifiers_info[model_name]['model_data']['y_test']

        # Retornando scores de treino e de teste
        train_scores = self.classifiers_info[model_name]['train_scores']
        test_scores = self.classifiers_info[model_name]['test_scores']

        # Plotando distribuição de scores
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        sns.kdeplot(train_scores[y_train == 1], ax=axs[0], label='y=1', shade=shade, color='darkslateblue')
        sns.kdeplot(train_scores[y_train == 0], ax=axs[0], label='y=0', shade=shade, color='crimson')
        sns.kdeplot(test_scores[y_test == 1], ax=axs[1], label='y=1', shade=shade, color='darkslateblue')
        sns.kdeplot(test_scores[y_test == 0], ax=axs[1], label='y=0', shade=shade, color='crimson')

        # Customizando plotagem
        format_spines(axs[0], right_border=False)
        format_spines(axs[1], right_border=False)
        axs[0].set_title('Score Distribution - Training Data', size=12, color='dimgrey')
        axs[1].set_title('Score Distribution - Testing Data', size=12, color='dimgrey')
        plt.suptitle(f'Score Distribution: a Probability Approach for {model_name}\n', size=14, color='black')
        plt.show()

    def plot_score_bins(self, model_name, bin_range):
        """
        Etapas:

        Argumentos:

        Retorno:
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.classifiers_info.keys())}')
            return None

        # Criando array de bins
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins))
                       if i > 0]

        # Retornando scores de treino e criando um DataFrame
        train_scores = self.classifiers_info[model_name]['train_scores']
        y_train = self.classifiers_info[model_name]['model_data']['y_train']
        df_train_scores = pd.DataFrame({})
        df_train_scores['scores'] = train_scores
        df_train_scores['target'] = y_train
        df_train_scores['faixa'] = pd.cut(train_scores, bins, labels=bins_labels)

        # Calculando distribuição por cada faixa - treino
        df_train_rate = pd.crosstab(df_train_scores['faixa'], df_train_scores['target'])
        df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis=0)

        # Retornando scores de teste e criando um DataFrame
        test_scores = self.classifiers_info[model_name]['test_scores']
        y_test = self.classifiers_info[model_name]['model_data']['y_test']
        df_test_scores = pd.DataFrame({})
        df_test_scores['scores'] = test_scores
        df_test_scores['target'] = y_test
        df_test_scores['faixa'] = pd.cut(test_scores, bins, labels=bins_labels)

        # Calculando distribuição por cada faixa - teste
        df_test_rate = pd.crosstab(df_test_scores['faixa'], df_test_scores['target'])
        df_test_percent = df_test_rate.div(df_test_rate.sum(1).astype(float), axis=0)

        # Definindo figura de plotagem
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Plotando gráficos de volumetria de cada classe por faixa
        for df_scores, ax in zip([df_train_scores, df_test_scores], [axs[0, 0], axs[0, 1]]):
            sns.countplot(x='faixa', data=df_scores, hue='target', ax=ax, palette=['darkslateblue', 'crimson'])
            AnnotateBars(n_dec=0, color='dimgrey').vertical(ax)
            ax.legend(loc='upper right')
            format_spines(ax, right_border=False)

        # Plotando percentual de representatividade de cada classe por faixa
        for df_percent, ax in zip([df_train_percent, df_test_percent], [axs[1, 0], axs[1, 1]]):
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
        axs[0, 0].set_title('Quantity of each Class by Range - Train', size=12, color='dimgrey')
        axs[0, 1].set_title('Quantity of each Class by Range - Test', size=12, color='dimgrey')
        axs[1, 0].set_title('Percentage of each Class by Range - Train', size=12, color='dimgrey')
        axs[1, 1].set_title('Percentage of each Class by Range - Test', size=12, color='dimgrey')
        plt.suptitle(f'Score Distribution by Range - {model_name}\n', size=14, color='black')
        plt.tight_layout()
        plt.show()

    def shap_analysis(self, model_name):
        """
        Parâmetros
        ----------
        classifiers: conjunto de classificadores em forma de dicionário [dict]
        X: array com os dados a serem utilizados no treinamento [np.array]
        y: array com o vetor target do modelo [np.array]

        Retorno
        -------
        None
        """

        # Retornando modelo a ser avaliado
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classificador {model_name} não foi treinado.')
            print(f'Opções possíveis: {list(self.classifiers_info.keys())}')
            return None

        # Applying shap approach
        explainer = shap.TreeExplainer(model)
        df_X_train_prep = pd.DataFrame(X_train_prep, columns=model_features)
        shap_values = explainer.shap_values(df_X_train_prep)

        # Plotting a summary plot using shap
        shap.summary_plot(shap_values[1], df_X_train_prep)