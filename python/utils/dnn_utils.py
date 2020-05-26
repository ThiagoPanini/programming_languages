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
"""

"""
--------------------------------------------
---------- IMPORTANDO BIBLIOTECAS ----------
--------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.viz_utils import *


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