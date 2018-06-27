import numpy as np


class RBF:

    pesos = np.zeros()

    def __init__(self, tam_amostra, num_centroides, num_classes):
        # tam_amostra = tamanho da amostra, número de dimensões das amostras que serão usadas na rede
        # num_centroides = quantidade de centroides usados na camada escondida = quantidade de neurônios utilizados na camada escondida
        # num_classes = quantidade de classes a serem classificadas pela rede = quantidade de neurônios utilizados na camada de saida

