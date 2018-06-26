
from parser import Parser
import numpy as np



parser = Parser()

base, labels, labels_nome = parser.get_base('../bases/faces95/')
autova, autove, media_treino = parser.eigenfaces_fit(base)
#base_lda = parser.lda(base, labels)

#equ = parser.img_equalize_histogram(img)

#img = parser.img_color2gray(img)

#res = np.hstack((img,equ)) #stacking images side-by-side
#parser.img_show(res)

#parser.img_write('../bases/equalize.png', res)

