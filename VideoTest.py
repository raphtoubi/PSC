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

"insérer les fonctions à l'endroit approprié"
def waveTransform2(wavelet,data) :
    res=pywt.dwt2(data,wavelet);
    return res[0][0][0];

def fourierTransform(data) :
    liste = fft(data);
    liste2=np.abs(liste);
    "enlève la composante continue"
    liste2[0]=0;
    """plt.plot(np.arange(0,n,1),liste2);
    plt.show();"""
    return liste2;
"fin des fonctions à insérer"

cap = cv2.VideoCapture(r'C:\Users\Alexian\Videos\Captures\GFAP8866.MP4')
"à initialiser"
m=300
i=0
tabfreqmax = []
data1=[];
data2=[];
data3=[];
fps=cap.get(cv2.CAP_PROP_FPS)
"fin de l'initialisation"
while(True):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)
        # write the flipped frame
        pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        pts = pts.reshape((-1,1,2))
        b,g,r = cv2.split(frame)
        "à insérer au moment où on chope les images (couleur = vert)"
        data1.append(waveTransform2('db2',g));
        data2.append(waveTransform2('db4',g));
        data3.append(waveTransform2('db7',g));
        "fin de l'insertion"
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    else:
        break
"à insérer une fois la vidéo finie"
n=len(data1)

while m+i<n :
    fourier = fourierTransform(data1[i:m+i]) + fourierTransform(data2[i:m+i]) + fourierTransform(data3[i:m+i])
    tabfreqmax.append(np.argmax(fourier[0:int(m/2)])* 60 / (m / fps))
    i=i+1
time = np.arange(len(tabfreqmax))
plt.plot(time, tabfreqmax)
plt.show()
"fin de l'insertion"
cap.release()
cv2.destroyAllWindows()




    