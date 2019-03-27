# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:10:44 2019

@author: Alexian
"""

import numpy as np
import cv2
import pywt;
import matplotlib.pyplot as plt;
from scipy.fftpack import fft;
from random import random;


def waveTransform2(wavelet,data) :
    res=pywt.dwt2(data,wavelet); "effectue la transformée en ondelettes sur data"
    return res[0][0][0];         "renvoie le premier coefficient"

def fourierTransform(data) :
    liste = fft(data);        "calcule la transformée de fourier (complexe)"
    liste2=np.abs(liste);     "calcule le module de chaque coefficient"
    liste2[0]=0;              "enlève la composante continue"
    """plt.plot(np.arange(0,n,1),liste2);
    plt.show();"""            "permet d'afficher la transformée de fourier si décommentée"
    return liste2;

def freqselect(listefreq, freqprec) :
    for i in range(np.size(listefreq)) :
        listefreq[i] = listefreq[i]/((np.abs(freqprec-i)+1)**3);
    sumcoeff=np.sum(listefreq);
    listefreq=listefreq/sumcoeff;
    r = random();
    somme=0;
    res=0;
    for i in range(np.size(listefreq)) :
        somme += listefreq[i];
        if(somme > r) :
            res=i;
            break;
    return res;

def main(nomDeLaVideo) :
    cap = cv2.VideoCapture(nomDeLaVideo);     "récupère la vidéo"
    m=300;            "nombre d'image considérée lors de la transformée de fourier"
    i=0;              "pointeur pour la transformée de fourier"
    tabfreqmax = [];  "stocke l'indice de la valeur maximale renvoyée par la transformée de fourier"
    data1=[];         "stocke les valeurs renvoyées par waveTransform2 pour une première onde"
    data2=[];         "deuxième onde"
    data3=[];         "troisième onde"
    fps=cap.get(cv2.CAP_PROP_FPS);  "récupère le nombre d'images par seconde"

    while(True):
        ret, frame = cap.read();                     "lit la vidéo image par image"
        "arrête la lecture si la vidéo est terminée"
        if ret==True:
            b,g,r = cv2.split(frame);                "sépare les couleurs de l'image"
            data1.append(waveTransform2('db2',g));   "rajoute la valeur calculée par waveTransform2 avec la première onde"
            data2.append(waveTransform2('db4',g));   "deuxième onde"
            data3.append(waveTransform2('db7',g));   "troisième onde"
            "arrête la lecture si l'utilisateur appuie sur q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    n=len(data1);                                    "récupère le nombre d'image de la vidéo"
    fourier = fourierTransform(data1[0:m]) + fourierTransform(data2[0:m]) + fourierTransform(data3[0:m]);
    i+=1;
    tabfreqmax.append(np.argmax(fourier[0:int(m/2)])* 60 / (m / fps));
    "tant que le pointeur + le nombre d'image nécessaire ne dépasse pas le nombre d'image total"
    while m+i<n :
        fourier = fourierTransform(data1[i:m+i]) + fourierTransform(data2[i:m+i]) + fourierTransform(data3[i:m+i]);     "somme les transformées de fourier sur les intervalles considérés"
        fourier[0]=0;
        fourier[1]=0;                   "évite l'attraction de forts coeffs à faibles fréquences quand l'image est très affectée"
        fourier[2]=0;
        tabfreqmax.append(freqselect(fourier[0:int(m/2)],tabfreqmax[-1]/(60 / (m / fps)))* 60 / (m / fps));         "calcule et ajoute l'indice du coefficient maximal de la transformée de fourier"
        i=i+1;                                                                  "met à jour le pointeur"
    time = np.arange(len(tabfreqmax));                                          "crée un tableau de 0 à len(tabfreqmax) avec un pas de 1"
    plt.plot(time, tabfreqmax);   "affiche l'indice du coefficient maximal de la transformée de fourier en fonction de l'image considérée"
    plt.xlabel('Image numéro');
    plt.ylabel('Fréquence estimée');
    plt.title('Fréquence cardiaque estimée pour chaque image');
    plt.show()
    cap.release()
    cv2.destroyAllWindows();

""" ne pas faire attention : stockage en cours
tabfreqmax.append(np.argmax(fourier[0:int(m/2)])* 60 / (m / fps));
vidéo pour les tests 'C:/Users/Alexian/Videos/Captures/GFAP8866.MP4' """
