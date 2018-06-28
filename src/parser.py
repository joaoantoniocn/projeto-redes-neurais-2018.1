import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from os import listdir
import numpy as np

# Parser para Pré Processamento dos dados
class Parser:

    def get_face(self, path):

        face_cascade = cv.CascadeClassifier('../haar/haarcascade_frontalface_default.xml')
        img = cv.imread(path)
        img2 = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #roi_gray = gray[y:y + h, x:x + w]
            img_face = img2[y:y + h, x:x + w]

        return img, img_face

    def img_show(self, img):

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def img_write(self, path, img):
        cv.imwrite(path, img)

    def img_equalize_histogram(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        equ = cv.equalizeHist(gray)

        return equ

    def img_color2gray(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return gray

    def img_read(self, path):
        img = cv.imread(path)

        return img

    def matrix2column(self, m):
        x, y = m.shape

        return np.reshape(m, (1, x*y))[0]


    def lda(self, base, labels):
        aux = LinearDiscriminantAnalysis()
        aux.fit(base, labels)

        return aux.transform(base)


    def ordena(self, autovalores, autovetores):

        autovalores_ordenados = autovalores
        autovetores_ordenados = autovetores

        for i in range(len(autovalores)):

            for j in range(i, len(autovalores)):

                if (autovalores_ordenados[i] < autovalores_ordenados[j]):
                    autovalores_temp = autovalores_ordenados[i]
                    autovetores_temp = autovetores_ordenados[:, i]

                    autovalores_ordenados[i] = autovalores_ordenados[j]
                    autovetores_ordenados[:, i] = autovetores_ordenados[:, j]

                    autovalores_ordenados[j] = autovalores_temp
                    autovetores_ordenados[:, j] = autovetores_temp

        return autovalores_ordenados, autovetores_ordenados

    def eigenfaces_fit(self, base, r):
        # base[amostras, características]
        # r = parametrô fracionario, quando r = 1 temos eigenfaces normal
        # quando 0 < r < 1, temos eigenfaces fracionário

        media = np.zeros(len(base[0]))



        # media
        for i in range(len(media)):
            media[i] = np.mean(base[:, i])

        # matrix de covariancia
        media = np.power(media, r)          # media fracionária
        baseT = np.power(base, r) - media   # base fracionária - media fracionária
        baseT = baseT.T # (cada coluna vira uma amostra)
        covariancia = np.matmul(baseT.T, baseT)
        covariancia = covariancia / len(base)

        # autovalores e autovetores
        autovalores, autovetores = np.linalg.eig(covariancia)

        # ordenando autovetores pelos maiores autovalores
        autovalores, autovetores = self.ordena(autovalores, autovetores)

        # expande autovetores
        autovetores_expandidos = np.zeros([len(baseT), len(base)])
        for i in range(len(autovalores)):
            e = autovetores[:, i]
            e = np.matmul(baseT, e)
            e = e / np.sqrt(len(autovalores) * autovalores[i])
            autovetores_expandidos[:, i] = e

        return autovalores, autovetores_expandidos, media

    def eigenfaces_transform(self, img, autovetores, media_treino, numero_features, r):
        # img é um vetor coluna que representa a amostra a ser projetada no Eigenfaces
        # numero_features indica em quantas dimensões queremos projetar a amostra
        # media_treino já vem fracionária se o eigenfaces foi treinado fracionário
        # r = parametro fracionário, serve para transformar a amostra em fracionária
        # se r = 1, temos eigenfaces normal
        # se 0 < r < 1, temos eigenfaces fracionário

        img_media = np.power(img, r) - media_treino
        E = autovetores[:, 0:numero_features]

        return np.matmul(E.T, img_media)

    def get_base(self, path):
        # O diretório passado em 'path' deve conter apenas pastas
        # onde cada uma das pastas corresponde a uma classe
        # dentro da pasta de cada classe deve ter n arquivos de imagem
        # cada imagem é uma amostra da base de dados
        # ---------------
        # Saída:
        # base[amostras, características]


        base = []
        labels = []
        labels_nome = []
        labels_binario = []

        classes = listdir(path)

        for classe in range(len(classes)):

            amostras = listdir(path + classes[classe] + str('/'))

            for amostra in range(len(amostras)):

                img = cv.imread(path + classes[classe] + str('/') + amostras[amostra])
                img = self.img_color2gray(img)

                base.append(self.matrix2column(img))
                labels.append(classe)
                labels_nome.append(classes[classe])
                labels_binario.append(np.array(list(np.binary_repr(classe, len(classes))), dtype=int))

        return np.asarray(base), labels, labels_nome, labels_binario

    def binariza(self, labels, qtd_classes):
        # transforma numero decimal em um vetor binario
        # Ex: 1 = 00001, 2 = 00010, 3 = 00011, 4 = 00100, 5 = 00101
        result = []

        for i in range(len(labels)):
            result.append(np.array(list(np.binary_repr(labels[i], qtd_classes)), dtype=int))

        return result

    def binariza2(self, labels, qtd_classes):
        # transforma um numero decimal em um vetor binario
        # onde só a posição do número será ativada
        # Ex: 2 = 00010, 3 = 00100, 4 = 01000, 5 = 10000

        result = []

        for i in range(len(labels)):
            aux = np.zeros(qtd_classes)
            aux[labels[i]] = 1
            result.append(aux)

        return result