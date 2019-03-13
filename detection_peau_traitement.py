




import logging
import numpy as np
import pywt;
from scipy.fftpack import fft;
import cv2
import skin_detector



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
faceCascade = cv2.CascadeClassifier('front.xml')
faceIndex = 0


def getFaceBox(image):
    global faceIndex
    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        return (0, 0, 0, 0)
    correspFace = None
    for f in faces:
        if f[2] > 100:
            cv2.imwrite('./faces/face-' + str(faceIndex) + '.png', image[f[1]:f[1] + f[3], f[0]:f[0] + f[2]])
            correspFace = f
            faceIndex += 1
            break
    if type(correspFace) == type(None):
        return (0, 0, 0, 0)
    return (correspFace[0], correspFace[1], correspFace[2], correspFace[3])



def waveTransform2(wavelet, data):
    res = pywt.dwt2(data, wavelet);
    return res[0][0][0];


def fourierTransform(data):
    liste = fft(data);
    liste2 = np.abs(liste);
    "enlève la composante continue"
    liste2[0] = 0;
    """plt.plot(np.arange(0,n,1),liste2);
    plt.show();"""
    return liste2;







##initialisation Alexian

m = 100
i = 0
tabfreqmax = [];
data1 = [];
data2 = [];
data3 = [];




def detecteur_de_peau(video, debug=False):

    #cam = cv2.VideoCapture('video-1540388616.mp4')

    cam = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640, 480))
    logging.info("press any key to exit")

    while True:

        ##DETECTION VISAGE RAPHAELLE
        ret, img_col = cam.read()
        if ret==True:
            bbox = getFaceBox(img_col)
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            crop_img = img_col[y:y + h, x:x + w]

            ##DETECTION PEAU ROXANE
            img_msk = skin_detector.process(crop_img)
            skin_detector.scripts.display('img_msk', img_msk)

            imb_comb = cv2.bitwise_and(crop_img, crop_img, mask=img_msk)
            skin_detector.scripts.display('img_skn', imb_comb)

            ##ALEXIAN
            pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            b, g, r = cv2.split(img_col)
            ##"à insérer au moment où on chope les images (couleur = vert)
            data1.append(waveTransform2('db2', g));
            ##data2.append(waveTransform2('db4', g));
            ##data3.append(waveTransform2('db7', g));

            waitkey = cv2.waitKey(5)
            if waitkey != -1:
                break
    n = len(data1)
    while m + i < n:

        fourier = fourierTransform(data1[i:m + i])## + fourierTransform(data2[i:m + i]) + fourierTransform(data3[i:m + i])
        tabfreqmax.append(np.argmax(fourier[0:int(m / 2)]))
        i = i + 1

    return out






n = len(data1)
while m + i < n:
    print("hello")
    fourier = fourierTransform(data1[i:m + i]) + fourierTransform(data2[i:m + i]) + fourierTransform(data3[i:m + i])
    tabfreqmax.append(np.argmax(fourier[0:int(m / 2)]))
    i = i + 1

cv2.destroyAllWindows()
