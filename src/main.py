
from parser import Parser
import cv2 as cv
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine


parser = Parser()


# ----- pré processamento ---
# pegando face e equalizando histograma
parser.base2face('../bases/faces95/', '../bases/faces95_faces/')


# base de teste
#data = load_wine()
#base_test = data.data
#labels_test = parser.binariza2(data.target, 3)
# -----------


#taxa_aprendizagem = 0.001
#epocas = 1000
#rbf = RBF([10, 10], taxa_aprendizagem, epocas)
#base, labels, labels_nome, labels_binario = parser.get_base('../bases/test/')
#rbf.fit(base, labels_binario)
#x = base[0]
#autova, autove, media_treino = parser.eigenfaces_fit(base, 1)
#base_lda = parser.lda(base, labels)

#equ = parser.img_equalize_histogram(img)

#img = parser.img_color2gray(img)

#res = np.hstack((img,equ)) #stacking images side-by-side
#parser.img_show(res)

#parser.img_write('../bases/equalize.png', res)

