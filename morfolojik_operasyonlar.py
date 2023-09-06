# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:56:05 2022

@author: Serpil ÖZGÜVEN
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktaralım

img = cv2.imread("datai_team.jpg",0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off"), plt.title("Orginal Img")


# erozyon
kernel = np.ones((5,5),dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result, cmap="gray"), plt.axis("off"), plt.title("Erozyon")


# genisleme dilation
result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result2, cmap="gray"), plt.axis("off"), plt.title("Genisleme")


#white noise
whiteNoise = np.random.randint(0,2, size = img.shape[:2])
whiteNoise = whiteNoise*255
plt.figure(), plt.imshow(whiteNoise, cmap= "gray"), plt.axis("off"), plt.title("White Noise")

#gürültüyü orjinal resim üzerine ekleyeceğim  ve noisel bir resim elde edeceğim
noise_img = whiteNoise + img
plt.figure(), plt.imshow(noise_img, cmap= "gray"), plt.axis("off"), plt.title("Img with White Noise")

# acılma(Noisei açılma yöntemiyle ortadan kaldıracağız )
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap= "gray"), plt.axis("off"), plt.title("Acilma")


# black noise(kapatmayi yapabilmek için gerekli)
# basit bir yöntemi var white noisedaki 255 i -255 yaparsaksiyah noise elde etmiş oluruz.
blackNoise = np.random.randint(0,2, size = img.shape[:2])
blackNoise = blackNoise*-255
plt.figure(), plt.imshow(blackNoise, cmap= "gray"), plt.axis("off"), plt.title("Black Noise")

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= -245] = 0
plt.figure(), plt.imshow(black_noise_img, cmap= "gray"), plt.axis("off"), plt.title("Black Noise Img")

# kapatma
closing = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure(), plt.imshow(closing, cmap= "gray"), plt.axis("off"), plt.title("Kapama")


# gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap= "gray"), plt.axis("off"), plt.title("Gradyan")





























