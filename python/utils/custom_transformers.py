"""
    Script criado para consolidar classes customizadas para automatização de Pipelines via Scikit-Learn
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

"""
--------------------------------------------
------ 1. PIPELINE PRÉ PROCESSAMENTO -------
--------------------------------------------
"""


# [PRE-PROCESSING] Classe para filtrar atributos de um DataFrame
class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, df):
        return self

    def transform(self, df):
        return df[self.features]


# [PRE-PROCESSING] Classe para dropar dados duplicados
class DropDuplicates(BaseEstimator, TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        return df.drop_duplicates()


# [PRE-PROCESSING] Classe para separar dados em treino e teste
class SplitData(BaseEstimator, TransformerMixin):

    def __init__(self, target, test_size=.20, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df):
        return self

    def transform(self, df):
        # Retornando conjuntos X e y
        X = df.drop(self.target, axis=1)
        y = df[self.target].values

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

# [PRE-PROCESSING] Classe para transformação de variável target
class TargetDefinition(BaseEstimator, TransformerMixin):
    """
    Class for building a rule for the target col and apply it to the dataset in order to
    make the data ready for feeding into an algorithm
    """
    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name may differ from target_col
        if self.target_col == self.new_target_name:
            print('[WARNING]')
            print(f'New target column named {self.new_target_name} may be different from raw one named {self.target_col}')

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Applying the new target rule based on positive class
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Dropping the old target column
        return df.drop(self.target_col, axis=1)
"""
--------------------------------------------
-------- 2. PIPELINE PROCESSAMENTO ---------
--------------------------------------------
"""


# [PROCESSING] Classe para aplicar processo de encoding nos dados
class DummiesEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, dummy_na=True):
        self.dummy_na = dummy_na

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Coletando variáveis
        self.cat_features_ori = list(X.columns)

        # Aplicando encoding com método get_dummies do pandas
        X_cat_dum = pd.get_dummies(X, dummy_na=self.dummy_na)

        # Juntando os datasets e eliminando colunas originais
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)
        self.features_after_encoding = list(X_dum.columns)

        return X_dum


# [PROCESSING] Classe para preencher dados nulos de um conjunto
class FillNullData(BaseEstimator, TransformerMixin):

    def __init__(self, cols_to_fill=None, fill_na=0):
        self.cols_to_fill = cols_to_fill
        self.fill_na = fill_na

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Verificando procedimento para colunas específicas
        if self.cols_to_fill is not None:
            return X.loc[:, self.cols_to_fill].fillna(self.fill_na)
        else:
            return X.fillna(self.fill_na)


"""
--------------------------------------------
--------- 3. TEXT PREP PIPELINES -----------
--------------------------------------------
"""

# [TEXT PREP] Classe para aplicar uma série de funções RegEx definidas em um dicionário
class ApplyRegex(BaseEstimator, TransformerMixin):

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying all regex functions in the regex_transformers dictionary
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)

        return X


# [TEXT PREP] Classe para aplicar a remoção de stopwords em um corpus
class StopWordsRemoval(BaseEstimator, TransformerMixin):

    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]


# [TEXT PREP] Classe para aplicar o processo de stemming em um corpus
class StemmingProcess(BaseEstimator, TransformerMixin):

    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]


# [TEXT PREP] Classe para extração de features de um corpus (vocabulário / bag of words / TF-IDF)
class TextFeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.vectorizer.fit_transform(X).toarray()


"""
--------------------------------------------
------------ 4. FUNÇÕES ÚTEIS --------------
--------------------------------------------
"""

# Leitura de base de dados de forma otimizada
def import_data(path, sep=',', optimized=True, n_lines=50, encoding='ANSI', verbose=True):
    """
    Etapas:
        1. leitura de primeiras linhas do dataset
        2. retorno de

    Argumentos:
        path -- caminho onde o dataset está localizado [string]
        sep -- separador de colunas da base [string, default: ',']
        optimized -- indicador de leitura de arquivo de forma otimizada [bool, default: True]
        n_lines -- quantidade de linhas a serem lidas no processo de otimização [int, default: 50]

    Retorno:
    """

    # Verificando indicador de redução de memória RAM
    if optimized:
        # Lendo primeiras linhas do dataset
        df_raw = pd.read_csv(path, sep=sep, nrows=n_lines, encoding=encoding)
        start_mem = df_raw.memory_usage().sum() / 1024 ** 2

        # Retornando atributos elegíveis à otimização
        float64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'float64']
        int64_cols = [col for col, dtype in df_raw.dtypes.items() if dtype == 'int64']
        total_opt = len(float64_cols) + len(int64_cols)
        if verbose:
            print(f'O dataset possui {df_raw.shape[1]} colunas, das quais {total_opt} são elegíveis a otimização.\n')

        # Otimizando tipos primitivos: float64 para float32
        for col in float64_cols:
            df_raw[col] = df_raw[col].astype('float32')

        # Otimizando tipos primitivos: int64 para int32
        for col in int64_cols:
            df_raw[col] = df_raw[col].astype('int32')

        # Verificando ganho de memória
        if verbose:
            print('----------------------------------------------------')
            print(f'Memória RAM utilizada ({n_lines} linhas): {start_mem:.4f} MB')
            end_mem = df_raw.memory_usage().sum() / 1024 ** 2
            print(f'Memória RAM após otimização ({n_lines} linhas): {end_mem:.4f} MB')
            print('----------------------------------------------------')
            mem_reduction = 100 * (1 - (end_mem / start_mem))
            print(f'\nGanho de {mem_reduction:.2f}% em uso de memória!\n')

        # Criando objeto com os novos tipos primitivos
        dtypes = df_raw.dtypes
        col_names = dtypes.index
        types = [dtype.name for dtype in dtypes.values]
        column_types = dict(zip(col_names, types))

        try:
            return pd.read_csv(path, sep=sep, dtype=column_types, encoding=encoding)
        except ValueError as e1:
            # Erro durante a leitura dos dados com os tipos primitivos passados
            print(f'Erro de valor ao ler arquivo: {e1}')
            print('Dataset será lido sem otimização de tipos primitivos.')
            return pd.read_csv(path, sep=sep)
    else:
        # Leitura de DataFrame sem otimização
        return pd.read_csv(path, sep=sep)

# Separação de atributos numéricos e categóricos de um DataFrame
def split_cat_num_data(X):
    """
    Etapas:
        1. levantamento de atributos numéricos e categóricos do conjunto

    Argumentos:
        df -- conjunto de dados [pandas.DataFrame]

    Retorno:
        num_attribs, cat_attribs -- atributos numéricos e categóricos [list]
    """

    # Separando atributos por tipo primitivo
    num_attribs = [col for col, dtype in X.dtypes.items() if dtype != 'object']
    cat_attribs = [col for col, dtype in X.dtypes.items() if dtype == 'object']

    return num_attribs, cat_attribs

# Cálculo de dias úteis entre duas colunas de datas de um DataFrame
def calc_working_days(date_series1, date_series2, convert=True):
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

    # Definindo função auxiliar para tratar exceções dentro de uma list comprehension
    def handle_working_day_calc(d1, d2):
        try:
            date_diff = np.busday_count(d1, d2)
            return date_diff
        except:
            return np.NaN

    # Convertendo Series em dados temporais
    if convert:
        date_series1 = pd.to_datetime(date_series1).values.astype('datetime64[D]')
        date_series2 = pd.to_datetime(date_series2).values.astype('datetime64[D]')

    # Construindo lista com diferença de dias úteis entre duas datas
    wd_list = [handle_working_day_calc(d1, d2) for d1, d2 in zip(date_series1, date_series2)]

    return wd_list