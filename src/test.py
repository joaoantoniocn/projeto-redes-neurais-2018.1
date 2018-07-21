
from parser import Parser
import cv2 as cv
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from cross_fold_validation import CrossFoldValidation


parser = Parser()


data = load_iris()
base = data.data
#base_norm, base1, base2 = parser.normaliza(data.data, data.data, data.data)
labels = np.asarray(parser.binariza2(data.target))


# --------------------- Cria RBF
parser.print("Instanciando RBF...")
taxa_aprendizagem = 1
epocas = 2004
camadas = [9, 3]
rbf = RBF(camadas, taxa_aprendizagem, epocas)
parser.print("Camadas: " + str(camadas))
parser.print("taxa de aprendizzagem = " +str(taxa_aprendizagem))
parser.print("numero de epocas = " +str(epocas))
parser.print("RBF instanciada!")
# -----

    # --- normalizando base de dados
#    parser.print("Normalizando base de dados...")
#    treino_norm, teste_norm, validacao_norm = parser.normaliza(treino_lda, teste_lda, validacao_lda)
#    parser.print("Normalização da base de dados concluida! (treino, teste e validacao)")

    # --- Treina rede
parser.print("Iniciando treinamento da RBF...")
rbf.fit(base, base, labels, labels)
parser.print("Treinamento finalizado!")


parser.close_log()

