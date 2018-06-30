import random
import numpy as np
import pickle

class CrossFoldValidation:

    def __init__(self, base, labels, numero_folders):
        # base = [amostra][atributo]
        # labels = numerericos ex: 1, 2, 3...
        # numero_folders = quantidade de folders

        self.base = base
        self.labels = labels
        self.numero_folders = numero_folders



    def separar_classe(self):
        # separa a base de dados pela classe

        separated = {}
        for i in range(len(self.base)):
            vector = self.base[i]
            if (self.labels[i] not in separated):
                separated[self.labels[i]] = []
            separated[self.labels[i]].append(vector)
        return separated


    def gerar_folders(self, path):
        # separa base de dados em folders
        # grava folders gerados em arquivo no diretório 'path'
        # retorna uma matriz[folder][classe][amostra][atributo]

        base_separada = self.separar_classe()
        folders = {}            # folders[numero_folder][classe][amostra][atributo]
        amostras_por_classe = int(len(base_separada[0]) / self.numero_folders)


        # folder
        for folder in range(self.numero_folders):

            if (folder not in folders):
                folders[folder] = {}

            # classe
            for classe in range(len(base_separada)):

                if (classe not in folders[folder]):
                    folders[folder][classe] = []

                # amostra
                for i in range(amostras_por_classe):

                    indice = random.randint(0, len(base_separada[classe])-1)
                    amostra = base_separada[classe][indice]

                    # adicionando a amostra no folder
                    folders[folder][classe].append(np.ndarray.tolist(amostra))

                    # removendo a amostra da base separada
                    del base_separada[classe][indice]

        # gravando folder em arquivo
        if (path != []):
            arq = open(path, 'wb')
            #arq.write(str(folders))
            pickle.dump(folders, arq)
            arq.close()

        return folders


    def get_folders_from_file(self, path):
        # carrega folders de arquivo
        # carrega folders gerados pela função self.gerar_folders()

        arq = open(path,'rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
        folders = pickle.load(arq)#Ler a stream a partir do arquivo e reconstroi o objeto original.
        arq.close()

        return folders

    def separa_treino_teste(self, folders, indice_teste, indice_validacao):
        # folders[folder][classe][amostra][atributo]
        # indice_test = numero inteiro ex: 1, 2, 3...
        # indice_validacao = array com 2 posições. Ex: [4, 5] posições diferente do indice test
        # retorna

        treino = []
        teste = []
        validacao = []

        label_teste = []
        label_validacao = []
        label_treino = []

        for folder in range(len(folders)):
            for classe in range(len(folders[folder])):

                #if (classe not in treino):
                #    treino[classe] = []
                #    teste[classe] = []
                #    validacao[classe] = []

                for amostra in range(len(folders[folder][classe])):
                    if (folder == indice_teste):
                        teste.append(np.array(folders[folder][classe][amostra]))
                        label_teste.append(classe)
                    elif ((folder == indice_validacao[0]) or (folder == indice_validacao[1])):
                        validacao.append(np.array(folders[folder][classe][amostra]))
                        label_validacao.append(classe)
                    else:
                        treino.append(np.array(folders[folder][classe][amostra]))
                        label_treino.append(classe)

        return np.asarray(treino), np.asarray(teste), np.asarray(validacao), label_treino, label_teste, label_validacao

#    def separa_treino_teste(folders, indice_teste, remove_col, indice_validacao):
#
#        treino = {}
#        teste = {}
#        validacao = {}

#        if (remove_col == 1):
#            for folder in range(len(folders)):
#                for classe in range(len(folders[folder])):

#                    if (classe not in treino):
#                        treino[classe] = []
#                        teste[classe] = []
#                        validacao[classe] = []

#                    for amostra in range(len(folders[folder][classe])):
#                        if (folder == indice_teste):
                            # np.delete(a, 1, 0)
#                            teste[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))

#                        elif((folder == indice_validacao[0]) or (folder == indice_validacao[1])):
#                            validacao[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))
#                        else:
#                            treino[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))
#        else:
#            for folder in range(len(folders)):
#                for classe in range(len(folders[folder])):

#                    if (classe not in treino):
#                        treino[classe] = []
#                        teste[classe] = []
#                        validacao[classe] = []

#                    for amostra in range(len(folders[folder][classe])):
#                        if (folder == indice_teste):
#                            # np.delete(a, 1, 0)
#                            teste[classe].append(folders[folder][classe][amostra])
#                        elif ((folder == indice_validacao[0]) or (folder == indice_validacao[1])):
#                            validacao[classe].append(folders[folder][classe][amostra])
#                        else:
#                            treino[classe].append(folders[folder][classe][amostra])

#        return treino, teste, validacao