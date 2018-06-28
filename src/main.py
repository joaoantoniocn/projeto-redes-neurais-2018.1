
from parser import Parser
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris


parser = Parser()

# Iris
data = load_iris()
base_iris = data.data
labels_iris = parser.binariza(data.target, 3)
# -----------


taxa_aprendizagem = 0.001
epocas = 1000
rbf = RBF([10, 3], taxa_aprendizagem, epocas)
base, labels, labels_nome, labels_binario = parser.get_base('../bases/test/')
rbf.fit(base_iris, labels_iris)
x = base_iris[0]
#autova, autove, media_treino = parser.eigenfaces_fit(base, 1)
#base_lda = parser.lda(base, labels)

#equ = parser.img_equalize_histogram(img)

#img = parser.img_color2gray(img)

#res = np.hstack((img,equ)) #stacking images side-by-side
#parser.img_show(res)

#parser.img_write('../bases/equalize.png', res)

