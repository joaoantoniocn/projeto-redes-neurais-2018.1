import numpy as np
from sklearn.cluster import KMeans


class RBF:

    pesos   = {} # [centroide][nodo_saida]
    entrada = {} # [camada][nodo];  [camada = 0] -> gaussiana; [camada = 1] -> saida
    camadas = [] # [num_centroides, num_classes]
    centroides = []
    std = []
    erro = []   # erro da predição atual
    taxa_aprendizagem = []
    epoca_treinamento = []
    sse = [] # Sum of Squared ERROR, soma do erro quadrado para todas as amostras de treinamento

    def __init__(self, camadas, taxa_aprendizagem, epoca_treinamento):
        # camadas = um array onde [tam_amostra, num_centroides, num_classes]
        # num_centroides = quantidade de centroides usados na camada escondida = quantidade de neurônios utilizados na camada escondida
        # num_classes = quantidade de classes a serem classificadas pela rede = quantidade de neurônios utilizados na camada de saida

        self.camadas = camadas
        self.taxa_aprendizagem = taxa_aprendizagem
        self.epoca_treinamento = epoca_treinamento
        self.sse = 0

        # iniciando pesos
        for i in range(camadas[0]): # camada escondida
            self.pesos[i] = []

            for j in range(camadas[1]): # nodo saida
                self.pesos[i].append(np.random.uniform(0, 1))

        # iniciando as entradas
        for camada in range(len(camadas)):
            self.entrada[camada] = np.zeros(camadas[camada])

            #for i in range(camadas[camada]):
             #   self.entrada[camada].append(0)

        # iniciando Erro
        self.erro = np.zeros(camadas[1])

    def fit(self, base, labels_binario):

        # definindo centros
        kmeans = KMeans(n_clusters = self.camadas[0], random_state=0)
        kmeans.fit(base)
        self.centroides = kmeans.cluster_centers_

        # definindo sigma
        self.calcula_sigma()

        # ------------------------
        # Treinando pesos da camada de saída
        # ------------------------

        # passando por todas as amostras de treinamento
        for epocas in range(self.epoca_treinamento):
            for i in range(len(base)):
                self.erro = labels_binario[i] - self.predict(base[i])

                if( np.sum(self.erro) != 0):
                    self.atualiza_pesos()

            print(self.calcula_sse(base, labels_binario))

    def atualiza_pesos(self):

        for unidade_escondida in range(self.camadas[0]):
            for unidade_saida in range(self.camadas[1]):
                self.pesos[unidade_escondida][unidade_saida] = self.pesos[unidade_escondida][unidade_saida] + self.taxa_aprendizagem * self.erro[unidade_saida] * self.entrada[0][unidade_escondida]

    def calcula_sse(self, base, labels_binario):
        result = 0

        for i in range(len(base)):
            erro = labels_binario[i] - self.predict(base[i])
            result = result + np.power(np.sum(erro), 2)

        self.sse = result
        return result
    def calcula_sigma(self):
        # definindo sigma [distancia media entre os centros mais próximos
        dist_sum = 0
        for centro1 in range(len(self.centroides)):

            for centro2 in range(len(self.centroides)):

                if (centro2 == 0):
                    if (centro1 != 0):
                        dist_menor = np.linalg.norm(self.centroides[centro1] - self.centroides[centro2])
                    else:
                        dist_menor = np.linalg.norm(self.centroides[centro1] - self.centroides[1])

                if (centro1 != centro2):
                    dist = np.linalg.norm(self.centroides[centro1] - self.centroides[centro2])
                    if (dist < dist_menor):
                        dist_menor = dist

            dist_sum = dist_sum + dist_menor
        self.std = (dist_sum / self.camadas[1])

    def gaussiana(self, x, centroide):

        dist = np.power(np.linalg.norm(x - centroide), 2)
        campo = 2*np.power(self.std, 2)

        return np.exp(-(dist/campo))

    def net(self, unidade_saida):
        sum = 0

        for nodo_escondido in range(self.camadas[0]): # unidade escondida
            sum += self.pesos[nodo_escondido][unidade_saida] * self.entrada[0][nodo_escondido]

        self.entrada[1][unidade_saida] = self.ativacao(sum)

    def ativacao(self, x):
        result = 0

        if (x > 0):
            result = 1

        return result
    def predict(self, x):

        # camada escondida
        for unidade_escondida in range(self.camadas[0]):
            self.entrada[0][unidade_escondida] = self.gaussiana(x, self.centroides[unidade_escondida])

        # camada saida
        for unidade_saida in range(self.camadas[1]):
            self.net(unidade_saida)

        return self.entrada[1]