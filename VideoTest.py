import numpy as np
import cv2
import pywt;
import matplotlib.pyplot as plt;
from scipy.fftpack import fft;


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


cap = cv2.VideoCapture(r'C:\Users\Alexian\Videos\Captures\GFAP8866.MP4');     "récupère la vidéo"

m=300;            #nombre d'image considérée lors de la transformée de fourier"
i=0;              #pointeur pour la transformée de fourier"
tabfreqmax = [];  #stocke l'indice de la valeur maximale renvoyée par la transformée de fourier"
data1=[];         #stocke les valeurs renvoyées par waveTransform2 pour une première onde"
data2=[];         #deuxième onde"
data3=[];         #troisième onde"
fps=cap.get(cv2.CAP_PROP_FPS);  #récupère le nombre d'images par seconde"

while(True):
    #lit la vidéo image par image:
    ret, frame = cap.read();                     
    #arrête la lecture si la vidéo est terminée
    if ret==True:
        b,g,r = cv2.split(frame);                #sépare les couleurs de l'image
        data1.append(waveTransform2('db2',g));   #rajoute la valeur calculée par waveTransform2 avec la première onde
        data2.append(waveTransform2('db4',g));   #deuxième onde
        data3.append(waveTransform2('db7',g));   #troisième onde
        #arrête la lecture si l'utilisateur appuie sur q:
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    else:
        break
        
#récupère le nombre d'image de la vidéo    
n=len(data1);

#tant que le pointeur + le nombre d'image nécessaire ne dépasse pas le nombre d'image total
while m+i<n :
    fourier = fourierTransform(data1[i:m+i]) + fourierTransform(data2[i:m+i]) + fourierTransform(data3[i:m+i]);     #somme les transformées de fourier sur les intervalles considérés
    tabfreqmax.append(np.argmax(fourier[0:int(m/2)])* 60 / (m / fps));         #calcule et ajoute l'indice du coefficient maximal de la transformée de fourier
    i=i+1;                                                                     #met à jour le pointeur
time = np.arange(len(tabfreqmax));                                             #crée un tableau de 0 à len(tabfreqmax) avec un pas de 1
plt.plot(time, tabfreqmax);   #affiche l'indice du coefficient maximal de la transformée de fourier en fonction de l'image considérée
plt.xlabel('Image numéro');
plt.ylabel('Fréquence estimée');
plt.title('Fréquence cardiaque estimée pour chaque image');
plt.show()
cap.release()
cv2.destroyAllWindows()




    
