
from parser import Parser
import numpy as np
from rbf import RBF



parser = Parser()
rbf = RBF([10, 10])
base, labels, labels_nome, labels_binario = parser.get_base('../bases/test/')
rbf.fit(base, labels_binario)
x = base[0]
#autova, autove, media_treino = parser.eigenfaces_fit(base, 1)
#base_lda = parser.lda(base, labels)

#equ = parser.img_equalize_histogram(img)

#img = parser.img_color2gray(img)

#res = np.hstack((img,equ)) #stacking images side-by-side
#parser.img_show(res)

#parser.img_write('../bases/equalize.png', res)

