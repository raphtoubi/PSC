# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:27:45 2019

@author: Alexian
"""
"Cas 2D"
import pywt;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.fftpack import fft;
import random;
"construit une matrice ligne dont les n valeurs sont un sinus de fréquence freq"
def sinMatrice(freq, n) :
    liste=np.zeros(n);
    for i in range(n) :
        liste[i]=np.sin(2*np.pi*freq*i/n);
    return liste;
"construit une matrice ligne dont les n valeurs sont un cosinus de fréquence freq"
def cosMatrice(freq,n) :
    liste = np.arange(0,n,1);
    liste=liste*(2*np.pi*freq/n);
    liste=np.cos(liste);
    return liste;
"construit une matrice n dont les valeurs sont comprises entre 0 et max (bruit)"
def randMatriceLigne(n,max=1):
    liste=np.zeros(n);
    for i in range (n) :
        liste[i] = random.random()*max-max/2;
    return liste;
"construit une matrice n,m dont les valeurs sont comprises entre 0 et max (bruit)"
def randMatrice(n,m,max=1):
    liste=np.zeros((n,m));
    for i in range (n):
        for j in range(m) :
            liste[i][j] = random.random()*max-max/2;
    return liste;
"construit une matrice n,m périodique de période T"
def periodMatrice(T,t,n,m,freq=1,valeur=1):
    liste=np.zeros((n,m));
    for i in range (int(n/T)):
        for j in range(int(m/T)) :
            liste[i*T][j*T]=valeur*np.sin(2*np.pi*freq*t);
    return liste;
"effectue une transformée de Fourier sur une matrice ligne data"
def fourierTransform(data) :
    n=data.size;
    liste = fft(data);
    liste2=np.abs(liste);
    "enlève la composante continue"
    liste2[0]=0;
    """plt.plot(np.arange(0,n,1),liste2);
    plt.show();"""
    return liste2;
def waveTransform(wavelet,T,t,n,m,freq=1,valeur=1,max=1) :
    liste1=randMatrice(n,m,max);
    liste2=periodMatrice(T,t,n,m,freq,valeur);
    "création d'une matrice bruitée"
    liste=liste1+liste2;
    res = pywt.dwt2(liste,wavelet);
    "plt.imshow(res[0]);"
    return res[0][0][0];
def waveTransform2(wavelet,data) :
    res=pywt.dwt2(data,wavelet);
    return res[0][0][0];
T=2;                "période spaciale de la matrice périodique"
n=100;              "nombre de points de l'échantillon"
taille=60;          "taille des matrices"
max=5;              "randMatrice a des coeffs entre 0 et max"
freq=10;            "fréquence temporelle voulue pour la matrice périodique"
data1=np.zeros(n);
data2=np.zeros(n);
data3=np.zeros(n);
fourier = np.zeros(n);
m=1
tabfreqmax=[]
for j in range(m) :
    for i in range(n) :
        liste1=randMatrice(taille,taille,max);
        liste2=periodMatrice(T,i/n,taille,taille,freq);
        data=liste1+liste2
        data1[i]=waveTransform2('db2',data);
        """data2[i]=waveTransform2('db4',data);
        data3[i]=waveTransform2('db7',data);"""
    fourier = fourierTransform(data1);""" + fourierTransform(data2) + fourierTransform(data3)"""
    tabfreqmax.append(np.argmax(fourier[0:int(n/2)]))
plt.plot(np.arange(n),fourier);
"""time = np.arange(len(tabfreqmax))
plt.plot(time, tabfreqmax)
plt.show()"""
