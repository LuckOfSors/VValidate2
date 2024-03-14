import numpy as np
import cv2
import pdf2image
import os

from skimage.transform import radon

#pdf goes here
filename = 'C:/Users/volpe/OneDrive/Desktop/Python/Varpratap I797 (1).PDF'

# convert pdf to jpeg
images = pdf2image.convert_from_path(filename)

for x, img in enumerate(images):
    img.save('VVImages/output'+str(x)+'.jpg', 'JPEG')
    # load file, converting to grayscale
    img = cv2.imread('VVImages/output'+str(x)+'.jpg')
    print(img.shape)
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = I.shape
    # resize image to reduce processing time
    if (w > 640):
        I = cv2.resize(I, (640, int((h / w) * 640)))
    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

    # Do the radon transform
    sinogram = radon(I)
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    print('Rotation: {:.2f} degrees'.format(90 - rotation))

    # Rotate and save with the original resolution
    M = cv2.getRotationMatrix2D((w/2,h/2),90 - rotation,1)
    dst = cv2.warpAffine(img,M,(w,h))
    cv2.imwrite('VVImages/rotated'+str(x)+'.jpg', dst)


aSide = 'VVImages/rotated0.jpg'
bSide = 'VVImages/rotated1.jpg'



