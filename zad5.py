import cv2
import numpy as np
from matplotlib import pyplot as plt


def showImage(image, title):
    fix, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
    ax.imshow(image)
    ax.set_title(title)
    plt.show()


# Zadanie 5
image = cv2.imread("dane/mleczyk.jpg")
grayImage = cv2.imread("dane/malpa.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

showImage(image, "Original")
showImage(grayImage, "Obrazek")

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 4))
ax.hist(grayImage.ravel(), bins=256, range=(0, 255), color='black')
plt.show()

firstPoint = [int(firstPoint) for firstPoint in input("Podaj wspolrzedne pierwszego punktu: ").split(',')]
secondPoint = [int(secondPoint) for secondPoint in input("Podaj wspolrzedne drugiego punktu: ").split(',')]

fragmentImage = image[firstPoint[1]:secondPoint[1], firstPoint[0]:secondPoint[0]]

showImage(fragmentImage, "Fragment obrazka")

fileName = input("Podaj nazwe pliku do zapisu: ")
fileName = fileName + ".jpg"
fragmentImage = cv2.cvtColor(fragmentImage, cv2.COLOR_BGR2RGB)
cv2.imwrite("dane/" + fileName, fragmentImage)

# Zadanie 6.1

s = 2
greyImageChange = grayImage * s

showImage(greyImageChange, "Obrazek po mnozeniu przez " + str(s))

# Zadanie 6.2

imageFrom0To1 = cv2.imread('dane/malpa.jpg').astype(np.float32) / 255
arrayOfImage = np.array(imageFrom0To1)


def calculateContrast(pixel, m, e):
    return 1 / (1 + (m / pixel) ** e)


def makeContrast(image, newImage, m, e):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                newImage[i][j][k] = calculateContrast(image[i][j][k], m, e)
    return newImage


newImage = np.zeros(arrayOfImage.shape)
newImage = makeContrast(arrayOfImage, newImage, 0.45, 8)

showImage(newImage, "Obrazek po zmianie kontrastu")

#Zadanie 6.3

gamma = 0.2

gammaImage = arrayOfImage ** gamma

showImage(gammaImage, "Obrazek po zmianie gamma")

#Zadanie 7

gammaImage = gammaImage * 255

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 4))
ax.hist(gammaImage.ravel(), bins=256, range=(0, 255), color='black')
plt.show()

#Zadanie 8

noiseImageSaltPepper = cv2.imread("dane/zaszumienie_solpieprz.jpg")

showImage(noiseImageSaltPepper, "Obrazek zaszumiony")

for i in range(3, 12):
    if i%2 == 1:
        medianBlur = cv2.medianBlur(noiseImageSaltPepper, i)
        showImage(medianBlur, "Obrazek po medianowaniu zaszumionego")
        cv2.imwrite("obrazkizaszumienia/medianBlur" + str(i) + ".jpg", medianBlur)

for i in range(2, 12):
    if i%2 == 1:
        blur = cv2.blur(noiseImageSaltPepper, (i, i))
        showImage(blur, "Obrazek po medianowaniu zaszumionego")
        cv2.imwrite("obrazkizaszumienia/blur" + str(i) + ".jpg", blur)

#Zadanie 9

motomoto = cv2.imread("dane/motomoto.jpg")

motomoto = cv2.cvtColor(motomoto, cv2.COLOR_BGR2GRAY)

showImage(motomoto, "Obrazek motomoto")

plt.imshow(cv2.Sobel(motomoto, cv2.CV_64F, 1, 0, ksize=5), cmap='gray')
plt.show()

plt.imshow(cv2.Sobel(motomoto, cv2.CV_64F, 0, 1, ksize=5), cmap='gray')
plt.show()

plt.imshow(cv2.Laplacian(motomoto, cv2.CV_64F, ksize=5), cmap='gray')
plt.show()