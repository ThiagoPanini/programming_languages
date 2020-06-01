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


"""
--------------------------------------------
--------- 1. PREPARAÇÃO DE DADOS -----------
--------------------------------------------
"""


# Criando classe para consolidar principais passos de um DataPrep
class DataPrep():

    def drop_cols(self, df, cols):
        """
        Etapas:
            1. exclusão de colunas do conjunto de dados

        Argumentos:
            df -- DataFrame [pandas.DataFrame]
            cols -- colunas a serem eliminadas [list]

        Retorno:
            df -- DataFrame após exclusão das colunas
        """

        # Dropando colunas
        return df.drop(cols, axis=1)

    def transform_bool_data(self, df):
        """
        Etapas:
            1. levantamento de todas as colunas de tipo booleano do conjunto
            2. multiplicação por 1 para mapear True = 1 e False = 0
            3. validação da existência de colunas booleanas no dataset após transformação

        Argumentos:
            df -- conjunto de dados [pandas.DataFrame]

        Retorno:
            df -- conjunto de dados após transformação [pandas.DataFrame]
        """

        # Filtrando atributos booleano do dataset
        bool_cols = [col for col in df.columns if df[col].dtype == 'bool']

        # Aplicando: True = 1 e False = 0
        for c in bool_cols:
            df[c] = df[c] * 1

        # Validando função
        try:
            assert len([col for col in df.columns if df[col].dtype == 'bool']) == 0
        except AssertionError:
            print(f'Ainda existem variáveis booleanas no conjunto. Verificar função!')
            return None

        return df

    def data_overview(self, df, label_name, sort_by='qtd_null', thresh_percent_null=0, thresh_corr_label=0):
        """
        Etapas:
            1. levantamento de atributos com dados nulos no conjunto
            2. análise do tipo primitivo de cada atributo
            3. análise da quantidade de entradas em caso de atributos categóricos
            4. extração da correlação pearson com o target para cada atributo
            5. aplicação de regras definidas nos argumentos
            6. retorno do dataset de overview criado

        Argumentos:
            df -- DataFrame a ser analisado [pandas.DataFrame]
            label_name -- nome da variável target [string]
            sort_by -- coluna de ordenação do dataset de overview [string - default: 'qtd_null']
            thresh_percent_null -- filtro de dados nulos [int - default: 0]
            threh_corr_label -- filtro de correlação com o target [int - default: 0]

        Retorno
            df_overview -- dataet consolidado contendo análise das colunas [pandas.DataFrame]
        """

        # Criando DataFrame com informações de dados nulos
        df_null = pd.DataFrame(df.isnull().sum()).reset_index()
        df_null.columns = ['feature', 'qtd_null']
        df_null['percent_null'] = df_null['qtd_null'] / len(df)

        # Retornando tipo primitivo e qtd de entradas para os categóricos
        df_null['dtype'] = df_null['feature'].apply(lambda x: df[x].dtype)
        df_null['qtd_cat'] = [len(df[col].value_counts()) if df[col].dtype == 'object' else 0 for col in
                              df_null['feature'].values]

        # Extraindo informação de correlação com o target
        label_corr = pd.DataFrame(df.corr()[label_name])
        label_corr = label_corr.reset_index()
        label_corr.columns = ['feature', 'target_pearson_corr']

        # Unindo informações
        df_null_overview = df_null.merge(label_corr, how='left', on='feature')

        # Filtrando dados nulos de acordo com limiares
        df_null_overview.query('percent_null > @thresh_percent_null')
        df_null_overview.query('target_pearson_corr > @thresh_corr_label')

        # Ordenando DataFrame
        df_null_overview = df_null_overview.sort_values(by=sort_by, ascending=False)
        df_null_overview = df_null_overview.reset_index(drop=True)

        return df_null_overview

    def split_cat_num_data(self, X):
        """
        Etapas:
            1. levantamento de atributos numéricos e categóricos do conjunto
            2. separação de datasets numéricos e categóricos

        Argumentos:
            X -- conjunto de dados (apenas features) [pandas.DataFrame]

        Retorno:
            X_num, X_cat -- datasets numéricos e categóricos [pandas.DataFrame]
            num_attribs, cat_attribs -- atributos numéricos e categóricos [list]
        """

        # Separando atributos por tipo primitivo
        num_attribs = [col for col, dtype in X.dtypes.items() if dtype != 'object']
        cat_attribs = [col for col, dtype in X.dtypes.items() if dtype == 'object']

        # Indexando DataFrame
        X_num = X[num_attribs]
        X_cat = X[cat_attribs]

        return X_num, X_cat, num_attribs, cat_attribs

    def null_data_strategy(self, X_num, X_cat, num_strategy, cat_strategy, fill_na_num=0, fill_na_cat='Unknown'):
        """
        Etapas:

        Argumentos:

        Retorno:
        """

        # Validando se existem dados nulos
        num_null_qtd = X_num.isnull().sum()
        cat_null_qtd = X_cat.isnull().sum()

        # Tratando dados nulos em variáveis numéricas
        if num_strategy == 'fill':
            X_num_wo_null = X_num.fillna(fill_na_num)