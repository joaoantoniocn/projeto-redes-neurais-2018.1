import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from os import listdir

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

    def lda(self):
        aux = LinearDiscriminantAnalysis()

        return aux

    def get_base(self, path):
        # O diretório passado em 'path' deve conter apenas pastas
        # onde cada uma das pastas corresponde a uma classe
        # dentro da pasta de cada classe deve ter n arquivos de imagem
        # cada imagem é uma amostra da base de dados

        base = []
        labels = []
        labels_nome = []

        classes = listdir(path)

        for classe in range(len(classes)):

            amostras = listdir(path + classes[classe] + str('/'))

            for amostra in range(len(amostras)):

                img = cv.imread(path + classes[classe] + str('/') + amostras[amostra])
                img = self.img_color2gray(img)

                base.append(img)
                labels.append(classe)
                labels_nome.append(classes[classe])

        return base, labels, labels_nome
