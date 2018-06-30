
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

# carrega base de dados
parser.print("Carregando base de dados...")
base, labels, labels_nome, labels_binario = parser.get_base('../bases/CroppedYale_faces/')
parser.print("Base de dados carregada!")
# -----

# --------------------- Cria RBF
parser.print("Instanciando RBF...")
taxa_aprendizagem = 0.001
epocas = 2000
rbf = RBF([len(set(labels)), len(set(labels))], taxa_aprendizagem, epocas)
parser.print("taxa de aprendizzagem = " +str(taxa_aprendizagem))
parser.print("numero de epocas = " +str(epocas))
parser.print("RBF instanciada!")
# -----

# ----- Separa base de dados em folders
parser.print("Separando a base de dados em treino, teste e validação")
cross = CrossFoldValidation(base, labels, 10)
folders = cross.gerar_folders([])
treino, teste, validacao, label_treino, label_teste, label_validacao = cross.separa_treino_teste(folders, 1, [2, 3])
parser.print("Base de dados separada!")

# for aqui

# --- projetando base com eigenfaces
parser.print("Projetando eigenfaces...")
r = 1 # parametro fracionário
representatividade = 0.9 # representatividade da projeção
autova, autove, media_treino = parser.eigenfaces_fit(treino, r)
treino_eig = parser.eigenfaces_transform_base(treino, autove, autova, media_treino, r, representatividade)
parser.print("Base de treino projetada!")
validacao_eig = parser.eigenfaces_transform_base(validacao, autove, autova, media_treino, r, representatividade)
parser.print("Base de validacao projetada!")
teste_eig = parser.eigenfaces_transform_base(teste, autove, autova, media_treino, r, representatividade)
parser.print("Base de teste projetada!")
parser.print("Dimensão da base de dados original:" + str(treino.shape))
parser.print("Dimensão da base de dados projetada:" + str(treino_eig.shape))
parser.print("Base de dados projetada!")
# ---

# --- normalizando base de dados
parser.print("Normalizando base de dados...")
treino_norm, teste_norm, validacao_norm = parser.normaliza(treino_eig, teste_eig, validacao_eig)
parser.print("Normalização da base de dados concluida! (treino, teste e validacao)")

# --- Treina rede
parser.print("Iniciando treinamento da RBF...")
rbf.fit(treino_norm, validacao_norm, parser.binariza2(label_treino), parser.binariza2(label_validacao))
parser.print("Treinamento finalizado!")
# ---




#base_lda = parser.lda(base, labels)


