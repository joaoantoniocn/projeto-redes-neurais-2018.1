
from parser import Parser
import cv2 as cv
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine


parser = Parser()


# ----- pré processamento ---
# pegando face e equalizando histograma
#parser.base2face('../bases/faces95/', '../bases/faces95_faces/')
# --------

taxa_aprendizagem = 0.001
epocas = 1000
rbf = RBF([10, 10], taxa_aprendizagem, epocas)
base, labels, labels_nome, labels_binario = parser.get_base('../bases/test/')
#rbf.fit(base, labels_binario)

autova, autove, media_treino = parser.eigenfaces_fit(base, 1)
base_eig = parser.eigenfaces_transform_base(base, autove, autova, media_treino, 1, 0.9)
#base_lda = parser.lda(base, labels)


