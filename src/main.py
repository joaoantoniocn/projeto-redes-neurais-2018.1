
from parser import Parser
import cv2 as cv
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from cross_fold_validation import CrossFoldValidation


parser = Parser()


# ----- pré processamento ---
# pegando face e equalizando histograma
#parser.base2face('../bases/faces95/', '../bases/faces95_faces/')
# --------

# --------------------- Cria RBF
taxa_aprendizagem = 0.001
epocas = 1000
rbf = RBF([10, 10], taxa_aprendizagem, epocas)
# -----

# carrega base d edados
base, labels, labels_nome, labels_binario = parser.get_base('../bases/test/')
# -----

# for aqui

# ----- Separa base de dados em folders
cross = CrossFoldValidation(base, labels, 10)
folders = cross.gerar_folders([])
treino, teste, validacao, label_treino, label_teste, label_validacao = cross.separa_treino_teste(folders, 1, [2, 3])

# --- projetando base com eigenfaces
autova, autove, media_treino = parser.eigenfaces_fit(treino, 1)
treino_eig = parser.eigenfaces_transform_base(treino, autove, autova, media_treino, 1, 0.9)
validacao_eig = parser.eigenfaces_transform_base(validacao, autove, autova, media_treino, 1, 0.9)
# ---

# --- Treina rede
rbf.fit(treino_eig, validacao_eig, parser.binariza2(label_treino, 10), parser.binariza2(label_validacao, 10))
# ---




#base_lda = parser.lda(base, labels)


