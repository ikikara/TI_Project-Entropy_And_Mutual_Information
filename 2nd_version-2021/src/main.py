import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


from huffmancodec import *

#1
def histograma(fonte, alfabeto, nome):
    Ocorr=ocorrencias(fonte, alfabeto)
    y=np.arange(len(alfabeto))

    plt.bar(y, Ocorr, align='center', color='orange')
    plt.xticks(y, alfabeto)
    plt.xlabel('Alfabeto')
    plt.ylabel('Ocorrencias')
    plt.title(nome)
    plt.show()

def ocorrencias(fonte, alfabeto):
    nrOcor = [0] * len(alfabeto)
    f=fonte.tolist()

    for i in range(len(alfabeto)):
        nrOcor[i]=f.count(alfabeto[i])

    return nrOcor


#2
def entropia(fonte):
    H=0
    Alfabeto, Ocorr= np.unique(fonte, axis=0,return_counts=True)
    for i in range(len(Ocorr)):
        if(Ocorr[i]!=0):
            probabilidade=Ocorr[i]/len(fonte)
            H+=probabilidade*math.log2(probabilidade)

    return round(-H,4)


#3
def LerFich(nome):
    extensao = nome.split('.')[1]
    alfabeto=[]
    P=[]

    if (extensao == "wav"):
        data = wavfile.read(nome)
        if (len(data[1].shape) == 2):
            P = data[1][:, 1]
        elif (len(data[1].shape) == 1):
            P = data[1]
        else:
            print("Ficheiro WAV inválido")
            return -1
        alfabeto = np.arange(256)

    elif (extensao == "bmp"):
        img = mpimg.imread(nome)
        if (len(img.shape) == 2):
            P = img.flatten()
        elif (len(img.shape) == 3):
            P = (img[:, :, 0].flatten())
        alfabeto = np.arange(256)

    elif (extensao == "txt"):
        P = [ch for ch in open(nome, 'r').read() if
             ((ord(ch) >= 65 and ord(ch) <= 90) or (ord(ch) >= 97 and ord(ch) <= 122))]
        alfabeto = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        alfabeto=np.array(alfabeto)

    return np.array(P), alfabeto


#4
def huffman_variancia(fonte):
    codec = HuffmanCodec.from_data(fonte)
    symbols, lenghts = codec.get_code_len()
    BS=Ef=Ef_2=0
    Alfabeto, Ocorr= np.unique(fonte, axis=0,return_counts=True)

    for i in range(len(lenghts)):
        for j in fonte:
            if(j==symbols[i]):
                BS+=lenghts[i]

        Ef_2+=lenghts[i]**2 * Ocorr[i]/len(fonte)
        Ef+=lenghts[i] * Ocorr[i]/len(fonte)

    V = round(Ef_2 - Ef ** 2, 4)


    return round(BS/len(fonte),3), V


#5
def entropiaMelhorada(fonte):
    P_novo,alf  = Agrupamento2(fonte)
    H = entropia(P_novo)

    return round(H / 2, 3)

def Agrupamento2(fonte):
    if (len(fonte) % 2 != 0):
        fonte = fonte[:len(fonte) - 1]
        P_novo = np.reshape(fonte, ((int)((len(fonte) + 1) / 2), 2))
    else:
        P_novo = np.reshape(fonte, ((int)((len(fonte)) / 2), 2))

    alf=np.unique(P_novo, axis=0)

    return P_novo, alf



#6
#a)
def InforMutua(query, target, passo):
    if(len(query)>len(target)):
        print('ERRO: impossível executar esta função visto que a "len" da query é maior que a "len" do target')
        return -1

    passo=(int)(passo)
    infMutua=[0.0]*math.ceil((len(target)-len(query)+1)/passo)
    target=np.array(target)
    query=np.array(query)

    entropia_query=entropia(np.array(query))

    for i in range(len(infMutua)):
        lista=target[passo*i:passo*i+len(query)]
        query_lista=[[query[i],lista[i]] for i in range(len(lista))]

        infMutua[i]=round((entropia_query - entropia(np.array(query_lista)) + entropia(np.array(lista))),4)

    return infMutua


#b)
def Targets(query, target, target2, passo):
    plt.plot(InforMutua(query,target,passo), label="target01 - repeat.wav")
    plt.plot(InforMutua(query, target2, passo), label="target02 - repeatNoise.wav")

    print(InforMutua(query,target,passo))
    print(InforMutua(query, target2, passo))

    plt.xlabel('"Subtargets"')
    plt.ylabel('Informação Mutua')

    plt.legend()
    plt.show()


#c)
def Shazam(query, passo):
    Targets=["Song01.wav","Song02.wav","Song03.wav","Song04.wav","Song05.wav","Song06.wav","Song07.wav"]

    for i in range(7):
        fonteT2, alfabetoT2 = LerFich(Targets[i])
        inf = InforMutua(query, fonteT2, passo)

        print("A informação mutua máxima do ficheiro '" + Targets[i] + "' é", max(inf))

        plt.plot(inf, label=Targets[i])

    plt.xlabel('"Subtargets"')
    plt.ylabel('Informação Mutua')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    Fontes=["homer.bmp","homerBin.bmp","kid.bmp","guitarSolo.wav","english.txt"]

    #3), 4) e 5)
    print("EXERCICIO 3, 4 e 5 JUNTOS")
    for i in range(5):
        nome = Fontes[i]
        fonte, alfabeto = LerFich(nome)
        Bs, var = huffman_variancia(fonte)

        print(nome + ":")
        print("H(x)= ", entropia(fonte),)
        print("Bits por simbolo= ", Bs, "e Variância= ", var)
        print("H(x) (modelização usando Agrupamento 2 a 2)= ", entropiaMelhorada(fonte), "\n")
        histograma(fonte, alfabeto, nome)
        #-------------------------------------------------------------------------------------------------------------------

    print("EXERCICIO 6 - a)")
    #6) (a)
    query= [2, 6, 4, 10, 5, 9, 5, 8, 0, 8]
    target = [6, 8, 9, 7, 2, 4, 9, 9, 4, 9, 1, 4, 8, 0, 1, 2, 2, 6, 3, 2, 0, 7, 4, 9, 5, 4, 8, 5, 2, 7, 8, 0, 7, 4, 8, 5, 7, 4, 3, 2, 2, 7, 3, 5, 2, 7, 4, 9, 9, 6]
    alfabetoNN = [0,1,2,3,4,5,6,7,8,9,10]
    passo = 1

    print("Informação Mutua da 'query' sobre o 'target' segundo um passo de 1:\n", InforMutua(query, target, passo),"\n")
    #-------------------------------------------------------------------------------------------------------------------

    print("EXERCICIO 6 - b)\n")
    # 6) (b)
    nomeQ = "guitarSolo.wav"
    fonteQ, alfabetoQ = LerFich(nomeQ)

    nomeT = "target01 - repeat.wav"
    fonteT, alfabetoT = LerFich(nomeT)

    nomeT = "target02 - repeatNoise.wav"
    fonteT2, alfabetoT2 = LerFich(nomeT)

    Targets(fonteQ, fonteT, fonteT2, len(fonteQ) / 4)
    #-------------------------------------------------------------------------------------------------------------------

    print("EXERCICIO 6 - c)")
    #6) (c)
    Shazam(fonteQ,len(fonteQ)/ 4)