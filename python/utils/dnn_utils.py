"""
    Script criado para consolidar funções úteis utilizadas em treinamentos de Redes Neurais profundas, de acordo com
a seguinte estrutura:

    1. Transformação de Inputs (image2vector)
    2. Inicialização dos Parâmetros
    3. Forward Propagation
    4. Custo
    5. Backward Propagation
    6. Atualização dos Parâmetros
    7. Consolidação de conceitos
    8. Predição e Resultados

    Segunda Onda:
    2.1. Classe completa para treinamento de uma Rede Neural Profunda
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from utils.viz_utils import *
import tensorflow as tf


"""
--------------------------------------------
------ 1. UTILITY FUNCTIONS IN DNNS --------
--------------------------------------------
"""

def fetch_batch(X, y, epoch, batch_index, batch_size):
    """
    This function read the data and returns the corresponding batch for applying batch optimization using TF

    Parameters
    ----------
    :param X: input data for extracting the batch (features) [type: array]
    :param y: input data for extracting the batch (target) [type: array]
    :param epoch: epoch number of batch extraction [type: int]
    :param batch_index: index of batch to be extracted [type: int]
    :param batch_size: size of the batch [type: int]

    Returns
    -------
    :return: X_batch, y_batch: batch extracted from both X and y arrays
    """

    # Retornando parâmetros
    m = X.shape[0]
    n_batches = m // batch_size

    # Definindo semente aleatória
    np.random.seed(epoch * n_batches + batch_index)

    # Indexando mini-batches do conjunto total
    indices = np.random.randint(m, size=batch_size)
    X_batch = X[indices]
    y_batch = y[indices]

    return X_batch, y_batch


"""
--------------------------------------------
------- 1. TRANSFORMAÇÃO DOS INPUTS --------
--------------------------------------------
"""


# Função para transformação das imagens
def image2vector_normalized(image_array):
    """
    Etapas:
    1. transforma um array de imagens em um único vetor utilizado nos cálculos da rede
    2. normaliza as intensidades de pixel em cada uma das células desse vetor via broadcasting

    Argumentos:
    image_array -- numpy array com set de imagens a serem transformadas [np.array]

    Retorno:
    image_normalized -- vetor de dimensões (nx*ny*nz, m) normalizado onde:
                        nx, ny e nz -- dimensões da imagem original
                        m -- quantidade de imagens presentes no dataset
    """

    image_flatten = image_array.reshape(image_array.shape[0], -1).T
    image_normalized = image_flatten / 255.

    return image_normalized


"""
--------------------------------------------
----- 2. INICIALIZAÇÃO DOS PARÂMETROS ------
--------------------------------------------
"""


# Função para inicialização dos parâmetros
def params_initialization(layers_dims, init_type='Xavier', seed=False):
    """
    Etapas:
    1. inicialização randômica de todos os parâmetros da rede

    Argumentos:
    layers_dims -- lista com especificação das dimensões da rede neural a ser construída [list]
    init_type -- definição do tipo de inicialização a ser utilizada [string]:
                'Xavier' -- Xavier initialization
                'He' -- He initialization
                'Random' -- Random initialization
    seed -- flag para utilização de semente aleatória [bool]

    Retorno:
    params -- dicionário contendo todo o set de parâmetros cujas chaves são 'W1', 'b1', ..., 'Wl', 'bl', onde:
                'Wl' -- matriz de parâmetros da camada l da rede com dimensões (layers_dims[l], layers_dims[l-1])
                'bl' -- vetor de bias da camada l da rede com dimensões (layers_dims[l], 1)
    """

    # Verificação de argumentos da função
    if init_type.lower().strip().replace(' ', '') not in ['random', 'xavier', 'he']:
        raise ValueError(f'{init_type} não permitido para o argumento init_type! Utilize "Random", "Xavier" ou "He"')

    # Definição de semente aleatória
    if seed:
        np.random.seed(42)

    params = {}  # Dicionário para alocação dos parâmetros
    L = len(layers_dims)  # Quantidade de camadas da rede

    # Iterando sobre cada uma das l camadas
    for l in range(1, L):

        # Verificando tipo de inicialização para W
        if init_type.lower().strip().replace(' ', '') == 'random':
            params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])
        elif init_type.lower().strip().replace(' ', '') == 'xavier':
            params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
        elif init_type.lower().strip().replace(' ', '') == 'he':
            params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])

        # Inicializando parâmetros b (independe do tipo de inicialização)
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return params


"""
--------------------------------------------
--------- 3. FORWARD PROPAGATION -----------
--------------------------------------------
"""


# Função para cálculo linear da camada
def linear_forward(A, W, b):
    """
    Etapas:
    1. aplicação de fórmula para cálculo da parcela linear da ativação das camadas

    Argumentos:
    A -- features (no caso da primeira camada) ou ativação das camadas anteriores da rede
    W, b -- set de parâmetros da rede

    Retorno:
    Z, cache -- parcela linear da rede e cache para utilização futura
    """

    # Cálculo linear de ativação
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


# Função de ativação sigmoidal
def sigmoid(Z):
    """
    Etapas:
    1. aplicação da função sigmoidal para ativação dos inputs lineares
    2. armazenamento de cache da entrada Z

    Argumentos:
    Z -- vetor obtido após parcela linear da ativação (cálculos via python broadcasting)

    Retorno:
    A, cache -- ativação da camada referente a entrada fornecida e cache para utilização futura
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


# Função de ativação de tangente hiperbólica (tanh)
def tanh(Z):
    """
    Etapas:
    1. aplicação da função tanh para ativação dos inputs lineares
    2. armazenamento de cache da entrada Z

    Argumentos:
    Z -- vetor obtido após parcela linear da ativação (cálculos via python broadcasting)

    Retorno:
    A, cache -- ativação da camada referente a entrada fornecida e cache para utilização futura
    """

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    cache = Z

    return A, cache


# Função de ativação Rectified Linear Unit (ReLU)
def relu(Z):
    """
    Etapas:
    1. aplicação da função ReLU para ativação dos inputs lineares
    2. armazenamento de cache da entrada Z

    Argumentos:
    Z -- vetor obtido após parcela linear da ativação (cálculos via python broadcasting)

    Retorno:
    A, cache -- ativação da camada referente a entrada fornecida e cache para utilização futura
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


# Função para ativação completa da camada
def linear_activation_forward(A_prev, W, b, activation):
    """
    Etapas:
    1. ativação linear da camada
    2. aplicação de função de ativação para ativação não-linear da camada
    3. salvamento de entradas em cache para utilização futura

    Argumentos:
    A_prev -- features (no caso da primeira camada) ou ativação das camadas anteriores da rede
    W, b -- set de parâmetros da rede
    activation -- string que define a função de ativação a ser utilizada nos cálculos

    Retorno:
    A, activation_cache-- ativação da camada atual da rede neural e cache de ativação
    """

    # Validação do argumento de ativação
    if activation.lower().strip().replace(' ', '') not in ['sigmoid', 'tanh', 'relu']:
        raise ValueError(
            f'{activation} não permitido para o argumento activation! Escolhe entre "sigmoid", "tanh" ou "relu"')

    # Ativação da parcela linear
    Z, linear_cache = linear_forward(A_prev, W, b)

    # Ativação da parcela não-linear
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'tanh':
        A, activation_cache = tanh(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    # Armazenando cache dos cálculos
    cache = (linear_cache, activation_cache)

    return A, cache


# Função para ativação de todas as camadas da rede
def L_model_forward(X, params, hidden_act='relu', output_act='sigmoid'):
    # Validação do argumento de ativação
    if hidden_act.lower().strip().replace(' ', '') not in ['sigmoid', 'tanh', 'relu']:
        raise ValueError(
            f'{hidden_act} não permitido para o argumento activation! Escolhe entre "sigmoid", "tanh" ou "relu"')
    if output_act.lower().strip().replace(' ', '') not in ['sigmoid', 'tanh', 'relu']:
        raise ValueError(
            f'{output_act} não permitido para o argumento activation! Escolhe entre "sigmoid", "tanh" ou "relu"')

    # Definindo variáveis úteis
    caches = []
    A = X
    L = len(params) // 2

    # Laço para ativação das camadas oculta
    for l in range(1, L):
        # Retornando os parâmetros da camada
        W = params['W' + str(l)]
        b = params['b' + str(l)]

        # Realizando cálculo de ativação das camadas ocultas
        A_prev = A
        A, cache = linear_activation_forward(A_prev, W, b, activation=hidden_act)
        caches.append(cache)

    # Ativação da última camada da rede
    WL = params['W' + str(L)]
    bL = params['b' + str(L)]
    AL, cache = linear_activation_forward(A, WL, bL, activation=output_act)
    caches.append(cache)

    return AL, caches


"""
--------------------------------------------
-------- 4. CUSTO DA REDE NEURAL -----------
--------------------------------------------
"""


# Função para cálculo do custo de cross-entropia
def compute_cost(AL, Y):
    """
    Etapas:
    1. cálculo do custo utilizando a fórmula de cross entropia

    Argumentos:
    AL -- ativação da última camada da rede calculada via forward propagation
    Y -- array com a variável resposta do conjunto de dados

    Retorno
    cost -- custo da rede treinada
    """

    # Definindo variáveis
    m = Y.shape[1]

    # Calculando custo
    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)  # Garantia das dimensões corretas

    return cost


"""
--------------------------------------------
-------- 5. BACKWARD PROPAGATION -----------
--------------------------------------------
"""


# Função para calular a derivada na última camada da rede
def last_layer_backward(AL, Y):
    """
    Etapas:
    1. cálculo da derivada parcial dAL da última camada AL da rede

    Argumentos:
    AL -- ativação da última camada da rede (output da função sigmoidal)
    Y -- vetor de label com as variáveis resposta do modelo

    Retorno
    dAL -- derivada parcial dAL a respeito da função custo (entropia cruzada)
    """

    return np.divide(-Y, AL) + np.divide((1 - Y), (1 - AL))


# Definição do gradiente a respeito da unidade sigmoidal
def sigmoid_backward(dA, cache):
    """
    Etapas:
    1. implementação da derivada da função sigmoidal

    Arguments:
    dA -- vetor de pós ativação da rede (obtido logo após o forward propagation)
    cache -- cache de estruturas armazenadas para esta implementação

    Returns:
    dZ -- gradiente do custo a respeito do termo Z da rede
    """

    # Retornando cache
    Z = cache

    # Calculando derivada
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


# Definição do gradiente a respeito da unidade sigmoidal
def tanh_backward(dA, cache):
    """
    Etapas:
    1. implementação da derivada da função tangente hiperbólica

    Arguments:
    dA -- vetor de pós ativação da rede (obtido logo após o forward propagation)
    cache -- cache de estruturas armazenadas para esta implementação

    Returns:
    dZ -- gradiente do custo a respeito do termo Z da rede
    """

    # Retornando cache
    Z = cache

    # Calculando derivada
    t = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    dZ = dA * (1 - t ** 2)

    return dZ


# Definição do gradiente a respeito da unidade ReLU
def relu_backward(dA, cache):
    """
    Etapas:
    1. implementação da derivada da função de ativação ReLU

    Arguments:
    dA -- vetor de pós ativação da rede (obtido logo após o forward propagation)
    cache -- cache de estruturas armazenadas para esta implementação

    Returns:
    dZ -- gradiente do custo a respeito do termo Z da rede
    """

    # Retornando cache
    Z = cache
    dZ = np.array(dA, copy=True)  # Apenas convertendo dZ para o objeto correto

    # Quando z <= 0, também é preciso setar dz como 0
    dZ[Z <= 0] = 0

    return dZ


def linear_backward(dZ, cache):
    """
    etapas:
    1. cálculo das derivadas parciais dW, db e dA_prev a partir da derivada dZ já calculada

    args:
    dZ -- derivada parcial da função de ativação da camada da rede
    cache -- tupla de valores (A_prev, W, b) originada no forward propagation da respectiva camada

    retorno:
    dA_prev, dW, db -- derivadas parciais das parcelas lineares do cálculo
    """

    # Retornando parâmetros da estrutura de cache
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # Definindo derivadas
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    """
    etapas:
    1. implementação da derivada da função sigmoidal

    args:
    dA -- vetor de pós ativação da rede (obtido logo após o forward propagation)
    cache -- cache de estruturas armazenadas para esta implementação

    retorno:
    dZ -- gradiente do custo a respeito do termo Z da rede
    """

    # Retornando cache
    Z = cache

    # Calculando derivada
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def relu_backward(dA, cache):
    """
    etapas:
    1. implementação da derivada da função de ativação ReLU

    args:
    dA -- vetor de pós ativação da rede (obtido logo após o forward propagation)
    cache -- cache de estruturas armazenadas para esta implementação

    retorno:
    dZ -- gradiente do custo a respeito do termo Z da rede
    """

    # Retornando cache
    Z = cache
    dZ = np.array(dA, copy=True)  # Apenas convertendo dZ para o objeto correto

    # Quando z <= 0, também é preciso setar dz como 0
    dZ[Z <= 0] = 0

    return dZ


# Função para calcular as derivadas das funções de ativação
def activation_backward(dA, cache, activation):
    """
    Etapas:
    1. cálculo da derivada parcial dZ de acordo com a função de ativação utilizada na camada
    2. cálculo das derivadas parciais de cada um dos parâmetros, além da derivada da próxima (anterior) camada

    Argumentos:
    dA -- derivada da ativação da camada l
    cache -- tupla de valores (linear_cache, activation_cache) armazenados para a corre implementação do backprop
    activation -- função de ativação utilizada na camada

    Retorno:
    dZ -- derivada da parcela linear da camada a respeito da função custo
    """

    # Validação do argumento de ativação
    if activation.lower().strip().replace(' ', '') not in ['sigmoid', 'tanh', 'relu']:
        raise ValueError(
            f'{activation} não permitido para o argumento activation! Escolhe entre "sigmoid", "tanh" ou "relu"')

    # Verificação da função de ativação da camada
    if activation == 'relu':
        dZ = relu_backward(dA, cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, cache)
    elif activation == 'tanh':
        dZ = tanh_backward(dA, cache)

    return dZ


# Função para cálculo das derivadas dos parâmetros da rede
def linear_backward(dZ, cache):
    """
    Etapas:
    1. cálculo das derivadas parciais dW, db e dA_prev a partir da derivada dZ já calculada

    Argumentos:
    dZ -- derivada parcial da função de ativação da camada da rede
    cache -- tupla de valores (A_prev, W, b) originada no forward propagation da respectiva camada

    Retorno:
    dA_prev, dW, db -- derivadas parciais das parcelas lineares do cálculo
    """

    # Retornando parâmetros da estrutura de cache
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # Definindo derivadas
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# Função para backward pass completo
def linear_activation_backward(dA, cache, activation):
    """
    Etapas:
    1. cálculo da derivada parcial dZ de acordo com a função de ativação utilizada na camada
    2. cálculo das derivadas parciais de cada um dos parâmetros, além da derivada da próxima (anterior) camada

    Argumentos:
    dA -- derivada da ativação da camada l
    cache -- tupla de valores (linear_cache, activation_cache) armazenados para a corre implementação do backprop
    activation -- função de ativação utilizada na camada

    Retorno:
    dA_prev -- Gradiente do custo a respeito da ativação da camada anterior (l-1), mesmas dimensões que A_prev
    dW -- Gradiente do custo a respeito de W (camada atual l), mesmas dimensões de W
    db -- Gradiente do custo a respeito de b (camada atual l), mesmas dimensões de b
    """

    # Retornando caches
    linear_cache, activation_cache = cache

    # Derivada das funções de ativação
    dZ = activation_backward(dA, activation_cache, activation)

    # Derivada dos parâmetros da rede
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Etapas:
    1. cálculo da derivada da derivada da última camada dAL do modelo em respeito a função custo
    2. retorno dos caches obtidos após o forward propagation
    3. utilização da função linear_activation_backward para cálculo das derivadas dA_prev, dWl, dbl da última camada
    4. aplicação de laço de repetição para cálculo das derivadas das camadas subsequentes

    Argumentos:
    AL -- vetor de probabilidades obtido na última etapa da função L_model_forward()
    Y -- vetor contendo a variável resposta de treino

    Retorno:
    grads -- dicionário com os gradientes (derivadas parciais) de cada um dos parâmetros dW, db e das ativações dA
    """

    # Criando e configurando variáveis
    grads = {}
    L = len(caches)  # Número de camadas da rede
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Inicialização do processo de backpropagation
    dAL = last_layer_backward(AL, Y)

    # Coletando cache da penúltima camada e calculando derivadas
    current_cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation='sigmoid')

    # Laço de repetição para cálculos nas demais camadas
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache,
                                                                    activation='relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads


"""
--------------------------------------------
------ 6. ATUALIZAÇÃO DOS PARÂMETROS -------
--------------------------------------------
"""


# Função para atualização dos parâmetros
def update_params(params, grads, learning_rate):
    """
    Etapas:
    1. unpack do dicionário de gradientes para a respectiva camada
    2. unpack do dicionário de parâmetros para a respectiva camada
    3. atualização dos parâmetros e armazenamento em novo dicionário

    Argumentos:
    params -- dicionário de parâmetros da rede (construído na inicialização randômica)
    grads -- dicionário de gradientes da rede (construído após o passo de backpropagation)
    learning_rate -- hiperparâmetro da rede que define a magnitude do passo do Gradiente Descendente

    Retorno:
    params -- dicionário contendo os parâmetros atualizados
                parameters["W" + str(l)] = ...
                parameters["b" + str(l)] = ...
    """

    # Quantidade de camadas da rede (para laço de iteração)
    L = len(params) // 2

    # Atualização de parâmetros para cada camada
    for l in range(L):
        params['W' + str(l + 1)] = params['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        params['b' + str(l + 1)] = params['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return params


"""
--------------------------------------------
------- 7. UNIFICANDO TREINAMENTO ----------
--------------------------------------------
"""


# Criando função para treinamento de uma rede neural de L camadas
# Criando função para treinamento de uma rede neural de L camadas
def L_layer_model(X, Y, layers_dims, init_type='Xavier', learning_rate=0.0075, num_iter=3000, print_cost=False,
                  title=''):
    """
    Etapas:
    1. inicialização aleatória dos parâmetros da rede de acordo com as dimensões de layers_dims
    2. apicação do forward propagation para cálculo das ativações de todas as camadas
    3. cálculo do custo da função
    4. aplicação do backward propagation para cálculo dos gradientes de todas as camadas
    5. utilização dos gradientes para atualização dos parâmetros
    6. repetir o processo de acordo com a definição da variável num_iter

    Argumentos:
    X -- conjunto de dados de entrada (features do modelo)
    Y -- conjunto com variável resposta do modelo
    layers_dims -- lista contendo a quantidade de neurônios em cada camada da rede
    learning_rate -- hiperparâmetro alpha que define a magnitude do passo do gradiente descendente
    num_iter -- número de iterações utilizadas no treinamento da rede
    print_cost -- flag para plotagem gráfica da curva de custo ao longo do treinamento
    title -- título dado ao gráfico da curva de custo ao longo do treinamento

    Retorno:
    params -- parâmetros treinados da rede
    """

    # Definição de variáveis e inicialização dos parâmetros
    np.random.seed(1)
    costs = []
    params = params_initialization(layers_dims, init_type=init_type)

    # Laço de repetição para treinamento
    for i in range(num_iter):

        # Forward propagation
        AL, caches = L_model_forward(X, params)

        # Computando custo
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Atualizando parâmetros
        params = update_params(params, grads, learning_rate)

        # Plotando custo a cada 100 iterações
        if print_cost and i % 100 == 0:
            print(f'Custo após a iteração {i}: {cost}')
            costs.append(cost)

    # Evolução do custo ao longo do treinamento
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.squeeze(costs), color='navy')
    ax.set_ylabel('Custo J')
    ax.set_xlabel('Iteração')
    ax.set_title(title, color='dimgrey')
    format_spines(ax, right_border=False)

    # Medindo acurácia do modelo e plotando
    plt.show()

    return params


"""
--------------------------------------------
-------- 8. PREDIÇÃO E RESULTADOS ----------
--------------------------------------------
"""


def predict(X, y, parameters):
    """
    etapas:
    1. retorno dos parâmetros da rede e preparação de variáveis
    2. aplicação do forward propagation com os parâmetros já treinados
    3. comparação da última camada com o vetor y contendo a variável resposta

    args:
    X -- conjunto de treinamento
    parameters -- parâmetros da rede treinada

    retorno:
    p -- predições (camada AL) ada rede
    """

    # Retornando parâmetros
    m = X.shape[1]
    n = len(parameters) // 2  # número de camadas da rede neural
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # Convertendo vetor probas (AL) em 0s e 1s
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # Printando acurácia do modelo
    acc = np.sum((p == y)/m)
    print(f'Acurácia: {round(acc, 2)}')

    return p


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))


"""
-------------------------------------------------------------------------------------
-------- SEGUNDA ONDA 2.1. Classe para Treinamento de Rede Neural Profunda ----------
-------------------------------------------------------------------------------------
"""

# Função para reset do grafo
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Classe para treinamento de uma Rede Neural Profunda
class DNNTraining():

    def __init__(self, n_hidden_layers=5, n_neurons=50, activation=tf.nn.relu, learning_rate=0.01,
                 optimizer=tf.train.AdamOptimizer, random_state=None, n_epochs=100, batch_size=128):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.session = None

        # Resetando grafo default
        reset_graph()

    def construction_phase(self):
        """
        Etapas:
            1. definição de variáveis;
            2. definição de placeholders;
            3. definição das camadas da rede;
            4. definição da função custo;
            5. definição do otimizador e training op;
            6. critérios de performance da rede;
            7. nós para visualização no TensorBoard;
            8. nós de inicialização e salvamento do modelo.

        Argumentos:
            self

        Retorno:
            None
        """

        # Criando placeholders para X e y
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name='X')
            y = tf.placeholder(tf.int32, shape=(None), name='y')

        # Definindo nós para as camadas da rede
        previous_layer = X
        with tf.name_scope('dnn'):
            for layer in range(self.n_hidden_layers):
                hidden_layer = tf.layers.dense(previous_layer, self.n_neurons, activation=self.activation,
                                               name=f'hidden{layer + 1}')
                previous_layer = hidden_layer
            # Última camada: logits e probabilidades
            logits = tf.layers.dense(previous_layer, self.n_outputs, name='outputs')
            y_proba = tf.nn.softmax(logits, name="y_proba")

        # Definindo função custo
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name='loss')

        # Operação de treinamento da rede
        with tf.name_scope('train'):
            training_op = self.optimizer(learning_rate=self.learning_rate).minimize(loss)

        # Métricas de avaliação: acurácia
        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        # Nós de inicialização da rede
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Parâmetros para visualização no TensorBoard
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        root_logdir = 'tf_logs'
        logdir = f'{root_logdir}/mnist_run_{now}'
        loss_summary = tf.summary.scalar('loss', loss)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        # Salvando alguns nós em forma de atributos para serem acessados posteriormente
        self.X, self.y = X, y
        self.y_proba, self.loss = y_proba, loss
        self.training_op = training_op
        self.accuracy = accuracy
        self.init, self.saver = init, saver
        self.loss_summary, self.file_writer = loss_summary, file_writer

    # Definindo função para encerramento da sessão
    def close_session(self):
        if self.session:
            self.session.close()

    # Definindo função para leitura de dados em mini-batches
    def fetch_batch(self, X, y, epoch, batch_index, batch_size):
        """
        Etapas:
            1. leitura do conjunto de dados em diferentes mini-batches

        Argumentos:
            X -- conjunto de dados a ser fatiado
            y -- vetor target a ser fatiado
            epoch -- época do treinamento do algoritmo
            batch_index -- índice do mini-batch a ser lido do conjunto total
            batch_size -- tamanho do mini-batch em termos de número de registros

        Retorno:
            X_batch, y_batch -- conjuntos mini-batch de dados lidos a partir do conjunto total
        """

        # Retornando parâmetros
        m = X.shape[0]
        n_batches = m // batch_size

        # Definindo semente aleatória
        np.random.seed(epoch * n_batches + batch_index)

        # Indexando mini-batches do conjunto total
        indices = np.random.randint(m, size=batch_size)
        X_batch = X[indices]
        y_batch = y[indices]

        return X_batch, y_batch

    # Função responsável por plotar o custo de treinamento da rede
    def plot_loss_curve(self, costs):
        """
        Etapas:
            1. plotagem da curva de custo a longo das épocas de treinamento

        Argumentos:
            costs -- lista com custos acumulados durante o treinamento [list]

        Retorno:
            None
        """

        # Plotando custo
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.squeeze(costs), color='navy')
        format_spines(ax, right_border=False)
        ax.set_title('Neural Network Cost', color='dimgrey')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        plt.show()

    def execution_phase(self, plot_cost=True):
        """
        Etapas:
            1. definindo parâmetros de controle e inicializando sessão
            2. iterando por cada época de treino e por cada mini-batch do conjunto de dados
            3. rodando operação de treinamento no mini-batch
            4. avaliando métricas da rede e medindo custo
            5. plotando curva de custo ao longo das épocas

        Argumentos:
            plot_cost -- indicador booleano para plotagem da curva de custo [bool - default: True]

        Retorno:
            None
        """

        # Definindo variáveis de controle
        m_train = self.X_train.shape[0]
        n_batches = m_train // self.batch_size
        costs = []

        # Inicializando sessão
        self.session = tf.Session(graph=self.graph)
        with self.session.as_default() as sess:

            # Inicializando variáveis globais
            self.init.run()

            # Iterando sobre as épocas de treinamento
            print('\n--- Inicializando Treinamento da Rede Neural ---\n')
            for epoch in range(self.n_epochs):
                # Iterando sobre cada mini-batch
                for batch in range(n_batches):
                    X_batch, y_batch = self.fetch_batch(self.X_train, self.y_train, epoch, batch, self.batch_size)
                    batch_feed_dict = {self.X: X_batch, self.y: y_batch}

                    # Salvando status do modelo a cada T mini-batches
                    if batch % 10 == 0:
                        summary_loss_str = self.loss_summary.eval(feed_dict=batch_feed_dict)
                        step = epoch * n_batches + batch
                        self.file_writer.add_summary(summary_loss_str, step)

                    # Inicializando treinamento em cada mini-batch
                    sess.run(self.training_op, feed_dict=batch_feed_dict)

                # Métricas de performance a cada N épocas
                test_feed_dict = {self.X: self.X_test, self.y: self.y_test}
                if epoch % 10 == 0:
                    acc_train = round(float(self.accuracy.eval(feed_dict=batch_feed_dict)), 4)
                    acc_test = round(float(self.accuracy.eval(feed_dict=test_feed_dict)), 4)
                    print(f'Epoch: {epoch}, Training acc: {acc_train}, Test acc: {acc_test}')

                # Custo da rede
                cost = self.loss.eval(feed_dict=batch_feed_dict)
                costs.append(cost)

            # Finalizando FileWriter
            self.file_writer.close()

            # Plotando custo de treinamento da rede
            if plot_cost:
                print('\n--- Plotando Custo ao longo das Épocas de Treinamento ---\n')
                self.plot_loss_curve(costs)

    # Função para encapsular todas as tratativas da rede neural
    def fit(self, X_train, y_train, X_test, y_test, plot_cost=True):
        """
        Etapas:
            1. encerramento de qualquer sessão aberta
            2. configurando grafo como atributo da classe
            3. chamando fase de construção da rede
            4. chamando fase de execução dos cálculos da rede

        Argumentos:
            X_train, y_train -- conjuntos de treinamento (features e target)
            X_test, y_test -- conjuntos de validação (features e target)
            plot_cost -- indicador booleano para plotagem da curva de custo [bool - default: True]

        Retorno:
            None
        """

        # Configurando atributos da classe
        self.X_train, self.X_test = X_train.astype('float32'), X_test.astype('float32')
        self.y_train, self.y_test = y_train.astype('int32'), y_test.astype('int32')
        self.n_inputs = X_train.shape[1]
        self.classes = np.unique(y_train)
        self.n_outputs = len(self.classes)

        # Encerrando sessão
        self.close_session()

        # Definindo grafo default e chamando construção da rede
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.construction_phase()

        # Chamando fase de execução da rede
        self.execution_phase(plot_cost=plot_cost)