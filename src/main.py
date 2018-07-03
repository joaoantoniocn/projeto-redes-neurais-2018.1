
from parser import Parser
import cv2 as cv
import numpy as np
from rbf import RBF
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from cross_fold_validation import CrossFoldValidation


parser = Parser()


def separa_treino_teste(folders, i):
    parser.print("Separando a base de dados em treino, teste e validação")
    indice_1 = (i + 1) % len(folders)
    indice_2 = (i + 2) % len(folders)
    indice_validacao = [indice_1, indice_2]
    treino, teste, validacao, label_treino, label_teste, label_validacao = cross.separa_treino_teste(folders, i,
                                                                                                     indice_validacao)
    parser.print("Base de dados separada!")

    return treino, teste, validacao, label_treino, label_teste, label_validacao

def projeta_eigenfaces(treino, validacao, teste, r):
    parser.print("Projetando eigenfaces...")
    #r = 1  # parametro fracionário
    parser.print("r = " + str(r))
    representatividade = 0.9  # representatividade da projeção
    parser.print("Representatividade: " + str(representatividade))

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

    return treino_eig, validacao_eig, teste_eig

def projeta_lda(treino, validacao, teste, label_treino):

    parser.print("Calculando LDA...")
    lda = parser.lda_fit(treino, label_treino)

    parser.print("Projetando base de treino...")
    treino_lda = parser.lda_transform(lda, treino)
    parser.print("Base de treino projetada!")

    parser.print("Projetando base de validacao...")
    validacao_lda = parser.lda_transform(lda, validacao)
    parser.print("Base de validacao projetada!")

    parser.print("Projetando base de teste...")
    teste_lda = parser.lda_transform(lda, teste)
    parser.print("Base de teste projetada!")

    parser.print("Projeção do LDA finalizada!")

    return treino_lda, validacao_lda, teste_lda


# ----- pré processamento ---
# pegando face e equalizando histograma
#parser.base2face('../bases/faces95/', '../bases/faces95_faces/')
# --------

# carrega base de dados
parser.print("Carregando base de dados...")
path = '../bases/CroppedYale_faces/'
parser.print(path)
base, labels, labels_nome, labels_binario = parser.get_base(path)
parser.print("Base de dados carregada!")
parser.print(path)
# -----

# --------------------- Cria RBF
parser.print("Instanciando RBF...")
taxa_aprendizagem = 1
epocas = 2004
camadas = [len(set(labels)), len(set(labels))]
rbf = RBF(camadas, taxa_aprendizagem, epocas)
parser.print("Camadas: " + str(camadas))
parser.print("taxa de aprendizzagem = " +str(taxa_aprendizagem))
parser.print("numero de epocas = " +str(epocas))
parser.print("RBF instanciada!")
# -----

# ----- Separa base de dados em folders
parser.print("Dividindo a base em folders")
num_folds = 10
parser.print(str(num_folds) + " Cross Validation")
cross = CrossFoldValidation(base, labels, num_folds)
folders = cross.gerar_folders([])


# Rodando os num_folds validation
taxa_acerto_folds = []
for i in range(num_folds):

    # instanciando RBF novamente para inicializar os pesos e a taxa de aprendizagem inicial do fold
    rbf = RBF(camadas, taxa_aprendizagem, epocas)

    parser.print("----------------------")
    parser.print("----------------------")
    parser.print("Fold " + str(i))
    parser.print("----------------------")
    parser.print("----------------------")

    # --- Separando treino teste e validação
    treino, teste, validacao, label_treino, label_teste, label_validacao = separa_treino_teste(folders, i)
    # ----

    # --- projetando base com eigenfaces
    treino_eig, validacao_eig, teste_eig = projeta_eigenfaces(treino, validacao, teste, 0.1)
    #  ---

    # --- projetando lda
    #treino_lda, validacao_lda, teste_lda = projeta_lda(treino, validacao, teste, label_treino)

    # --- normalizando base de dados
    parser.print("Normalizando base de dados...")
    treino_norm, teste_norm, validacao_norm = parser.normaliza(treino_eig, teste_eig, validacao_eig)
    parser.print("Normalização da base de dados concluida! (treino, teste e validacao)")

    # --- Treina rede
    parser.print("Iniciando treinamento da RBF...")
    rbf.fit(treino_norm, validacao_norm, parser.binariza2(label_treino), parser.binariza2(label_validacao))
    parser.print("Treinamento finalizado!")

    # --- Calculando taxa de acerto no conjunto de teste
    parser.print("Calculando taxa de acerto no conjunto de teste")
    taxa_acerto = rbf.calcula_taxa_acerto(teste_norm, parser.binariza2(label_teste))
    parser.print("Taxa de acerto do fold " + str(i) + ": " + str(taxa_acerto))
    taxa_acerto_folds.append(taxa_acerto)
    # ---

parser.print("Taxa de acerto do Experimento: " + str(taxa_acerto_folds))
parser.print("Media da taxa de acerto: " + str(np.mean(taxa_acerto_folds)))
parser.print("Desvio Padrão: " + str(np.std(taxa_acerto_folds)))

#base_lda = parser.lda(base, labels)


parser.close_log()

