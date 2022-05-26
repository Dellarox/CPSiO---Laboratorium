import os
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import cv2
import scipy.signal as sig

sinus = lambda f, fs, N: np.sin(2 * np.pi * f * np.linspace(0, N / fs, N))
linspace = lambda start, stop, num: np.linspace(start, num / stop, num)

defautlImage = "image.png"

layout = [[sg.Text("""
Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów
        Kajetan Krasoń 252767, Kacper Małkowski 252724            """)],
          [sg.Button("Sygnały"), sg.Button("Obrazy"), sg.Button("Wyjdz")]]
window = sg.Window("Aplikacja CPSiO", layout)


def showPlot(x, y, title, x1, x2, xlabel, ylabel, log=False, yLim=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 5))
    if log:
        ax.semilogx(x, y)
    else:
        ax.plot(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axhline(y=0, color='k')
    ax.grid(True)
    plt.xlim(x1, x2)
    if yLim != None:
        plt.ylim(yLim[0], yLim[1])
    plt.show()


def showImage(image, title):
    fix, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
    ax.imshow(image)
    ax.set_title(title)
    plt.show()


def signal1(x, signal, frequency, xLabel, yLabel):
    window.close()
    configure = [[sg.Text("czas poczatkowy [s]"), sg.InputText(key="t1")],
                 [sg.Text("czas koncowy [s]"), sg.InputText(key="t2")],
                 [sg.Button("Generuj"), sg.Button("Wroc")]
                 ]
    showSignal = sg.Window("Sterowanie wykresem", configure)

    while True:
        event, values = showSignal.read()
        if event == "Wroc" or event == sg.WIN_CLOSED:
            plt.close()
            break
        if event == "Generuj":
            plt.close()
            if values["t1"] == "":
                t1 = 0
            else:
                t1 = float(values["t1"])

            if values["t2"] == "":
                t2 = len(signal) / frequency
            else:
                t2 = float(values["t2"])

            showPlot(x, signal, "Sygnal", t1, t2, xLabel, yLabel)
    showSignal.close()


def fastFourier(signal, samplingFreq, samples, y=None):
    signalHelp = abs(np.fft.fft(signal))
    signalHelp = signalHelp / 2
    freq = np.linspace(0, samplingFreq / 2, samples // 2)

    if y == None:
        showPlot(freq, signalHelp[:samples // 2], "FFT", 0, samplingFreq / 2, "Czestotliwosc [Hz]", "Amplituda")
    else:
        showPlot(freq, signalHelp[:samples // 2], "FFT", 0, samplingFreq / 2, "Czestotliwosc [Hz]", "Amplituda", yLim=y)


def invertedFastFourier(signal, linspaces, freq=False):
    signalFFT = np.fft.fft(signal)
    signalIFFT = np.fft.ifft(signalFFT).real

    if freq:
        showPlot(linspaces, signalIFFT, "IFFT", 0, len(signal) / 360, "Czas [s]", "Amplituda")
    else:
        showPlot(linspaces, signalIFFT, "IFFT", 0, 1, "Czas [s]", "Amplituda")


def signal2():
    samples = 65536
    signalFreq = 50
    samplingFreq = 1024

    # Zadanie 2.1

    x = linspace(0, samplingFreq, samples)
    signal = sinus(signalFreq, samplingFreq, samples)

    showPlot(x, signal, "Sygnal sinusoidalny", 0, 1, "Czas [s]", "Amplituda")
    # Zadanie 2.2

    fastFourier(signal, samplingFreq, samples)

    # Zadanie 2.3

    firstFreq = 50
    secondFreq = 60
    firstSig = sinus(firstFreq, samplingFreq, samples)
    secondSig = sinus(secondFreq, samplingFreq, samples)
    x = linspace(0, samplingFreq, samples)
    showPlot(x, firstSig + secondSig, "Polaczone sygnaly", 0, 1, "Czas [s]", "Amplituda")

    fastFourier(firstSig + secondSig, samplingFreq, samples)

    # Zadanie 2.4

    samplingFreq = 512
    signalFreq = 50
    signal = sinus(signalFreq, samplingFreq, samples)
    x = linspace(0, samplingFreq, samples)

    showPlot(x, signal, "Sygnal zmieniony", 0, 1, "Czas [s]", "Amplituda")

    fastFourier(signal, 512, 65536)

    # Zadanie 2.5

    invertedFastFourier(signal, x)

    pass


def signal3():
    signal = np.loadtxt('dane\ekg100.txt')

    freq = 360
    time = len(signal) / freq
    x = np.linspace(0, time, len(signal))

    showPlot(x, signal, "Sygnal ekg100", 0, time, "Czas [s]", "Amplituda")

    # Zadanie 3.2

    ekgFFT = np.fft.fft(signal)
    ekgFFT = np.abs(ekgFFT)
    ekgFFT = ekgFFT / 2
    freq = np.linspace(0, freq / 2, len(ekgFFT) // 2)

    showPlot(freq, ekgFFT[:len(ekgFFT) // 2], "FFT ekg100", 0, freq[len(freq) // 2], "Czestotliwosc [Hz]", "Amplituda",
             yLim=[-2, 5000])

    # Zadanie 3.3

    x = linspace(0, 360, 1300000)

    invertedFastFourier(signal, x, freq=True)

    pass


def signal4():
    signal = np.loadtxt("dane\ekg_noise.txt")
    freq = 360
    samples = len(signal)

    x = signal[:, 0]
    signal = signal[:, 1]

    # Zadanie 4.1

    showPlot(x, signal, "Sygnal z ekg", 0, samples / freq, "Czas [s]", "Amplituda")

    # Zadanie 4.2

    fastFourier(signal, freq, samples, y=[0, 50])

    # Zadanie 4.3

    _, __ = sig.cheby1(4, 1, 60, 'low', False, fs=360)
    w, h = sig.freqs(_, __)

    showPlot(w, 20 * np.log10(abs(h)), "Wykres filtru dolnoprzepustowego", 0, samples / freq, "Czestotliwosc",
             "Amplituda [dB]", log=True, yLim=[-39, -26])

    # Zadanie 4.4

    filteredSignal = sig.filtfilt(_, __, signal)
    showPlot(x, filteredSignal, "Sygnal po filtracji", 0, samples / freq, "Czas [s]", "Amplituda")

    # Zadanie 4.5

    noise = signal - filteredSignal
    showPlot(x, noise, "Szum", 0, samples / freq, "Czas [s]", "Amplituda")

    # Zadanie 4.6

    fastFourier(filteredSignal, freq, samples, y=[0, 50])

    # Zadanie 4.7

    fastFourier(noise, freq, samples, y=[0, 50])

    # Zadanie 4.8

    _, __ = sig.cheby1(4, 1, 5, 'highpass', False, fs=360)
    w, h = sig.freqs(_, __)

    showPlot(w, 20 * np.log10(abs(h)), "Wykres filtru górnoprzepustowego", 0, samples / freq, "Czestotliwosc",
             "Amplituda [dB]", log=True)

    # Zadanie 4.9

    filteredSignal = sig.filtfilt(_, __, signal)
    showPlot(x, filteredSignal, "Sygnal po filtracji", 0, samples / freq, "Czas [s]", "Amplituda")

    # Zadanie 4.10

    noise = signal - filteredSignal
    showPlot(x, noise, "Szum", 0, samples / freq, "Czas [s]", "Amplituda")

    # Zadanie 4.11

    fastFourier(filteredSignal, freq, samples, y=[0, 50])

    # Zadanie 4.12

    fastFourier(noise, freq, samples, y=[0, 50])

    pass


def grey(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(defautlImage, image)
    return image
    pass


def saveFragment(image):
    p1 = tk.simpledialog.askstring("Pierwszy punkt ", "Podaj współrzędne pierwszego punktu (,)")
    p2 = tk.simpledialog.askstring("Drugi punkt ", "Podaj współrzędne drugiego punktu (,)")

    p1 = p1.split(",")
    p2 = p2.split(",")
    p1[0] = int(p1[0])
    p1[1] = int(p1[1])
    p2[0] = int(p2[0])
    p2[1] = int(p2[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageFragment = image[p1[1]:p2[1], p1[0]:p2[0]]

    showImage(imageFragment, "Fragment obrazka")
    cv2.imwrite("dane/croppedImage.jpg", imageFragment)


def multiply(image):
    const = tk.simpledialog.askfloat("Stala", "Podaj wspolczynnik mnozenia: ")
    if const:
        image = image * const
        cv2.imwrite(defautlImage, image)
    return image


def calculateContrast(pixel, m, e):
    if pixel <= 0:
        return 0
    return 1 / (1 + (m / pixel) ** e)


def makeContrast(image, newImage, m, e):
    for i in range(image.shape[0]):
        for ii in range(image.shape[1]):
            for iii in range(image.shape[2]):
                newImage[i][ii][iii] = calculateContrast(image[i][ii][iii], m, e)
    return newImage


# naprawić
def contrast(image):
    m = 0.45
    e = 4
    m = tk.simpledialog.askfloat("m", "Podaj m: ")
    e = tk.simpledialog.askfloat("e", "Podaj e: ")
    if m and e:
        imageFrom0To1 = cv2.imread(defautlImage).astype(np.float32) / 255
        arrayOfImage = np.array(imageFrom0To1)
        newImage = np.zeros(arrayOfImage.shape)
        newImage = makeContrast(arrayOfImage, newImage, m, e)
        newImage = newImage * 255
        newImage = newImage.astype(np.uint8)
        cv2.imwrite(defautlImage, newImage)
        return newImage
    return image


def gamma(image):
    imageFrom0To1 = cv2.imread(defautlImage).astype(np.float32) / 255
    arrayOfImage = np.array(imageFrom0To1)
    gamma = tk.simpledialog.askfloat("Gamma", "Podaj wspolczynnik gamma: ")
    if gamma:
        image = arrayOfImage ** (gamma)
        image = image * 255
        cv2.imwrite(defautlImage, image)
    return image


def histogram(image):
    imageFrom0To1 = cv2.imread(defautlImage).astype(np.float32)
    arrayOfImage = np.array(imageFrom0To1)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 4))
    ax.hist(arrayOfImage.ravel(), bins=256, range=(0, 255), color='black')
    plt.show()


def nonLinearMedianFilter(image):
    kernel = 5
    kernel = tk.simpledialog.askinteger("Wielkosc filtra", "Podaj wielkosc filtra: ")
    if kernel % 2 == 1:
        medianBlur = cv2.medianBlur(image, kernel)
        cv2.imwrite(defautlImage, medianBlur)
        return medianBlur
    messagebox.showerror("Błąd", "Wielkość filtra musi być nieparzysta")
    return image


def linearAvgFilter(image):
    kernel = 5
    kernel = tk.simpledialog.askinteger("Wielkosc filtra", "Podaj wielkosc filtra: ")
    if kernel % 2 == 1:
        avgBlur = cv2.medianBlur(image, kernel)
        cv2.imwrite(defautlImage, avgBlur)
        return avgBlur
    messagebox.showerror("Błąd", "Wielkość filtra musi być nieparzysta")
    return image


def sobelHorizontal(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    plt.imshow(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5), cmap='gray')
    plt.savefig(defautlImage)


def sobelVertical(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    plt.imshow(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5), cmap='gray')
    plt.savefig(defautlImage)


def laplacian(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
    plt.imshow(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5), cmap='gray')
    plt.savefig(defautlImage)


def save(image):
    cv2.imwrite("dane/image.jpg", image)


while True:
    event, values = window.read()
    if event == "Wyjdz" or event == sg.WIN_CLOSED:
        break

    if event == "Sygnały":
        window.close()
        signalLayout = [[sg.Button("cw.1"), sg.Button("cw.2"), sg.Button("cw.3"), sg.Button("cw.4")],
                        [sg.Button("Wroc")]]
        window = sg.Window("Aplikacja CPSiO", signalLayout)

        while True:
            event, values = window.read()
            if event == "cw.1":
                filename = askopenfilename()
                if filename == "":
                    messagebox.showerror("Błąd", "Nie wybrano pliku")
                    break
                signal = np.loadtxt(filename)

                if filename.split("/")[-1] == "ekg_noise.txt":
                    x = signal[:, 0]
                    signal = signal[:, 1]
                    f = 360
                elif filename.split("/")[-1] == "ekg100.txt":
                    f = 360
                    x = np.linspace(0, len(signal) / f, len(signal))
                elif filename.split("/")[-1] == "ekg1.txt":
                    f = 1000
                    x = np.linspace(0, len(signal) / f, len(signal))

                signal1(x, signal, f, "Próbki", "Amplituda")
            if event == "cw.2":
                signal2()
            if event == "cw.3":
                signal3()
            if event == "cw.4":
                signal4()
            if event == "Wroc" or event == sg.WIN_CLOSED:
                break

    if event == "Obrazy":
        filename = askopenfilename()
        if filename == "":
            messagebox.showerror("Błąd", "Nie wybrano pliku")
            break
        image = cv2.imread(filename)
        cv2.imwrite(defautlImage, image)
        window.close()
        imageLayout = [[sg.Image(filename=defautlImage, key="image")],
                       [[sg.Button("szarosc"), sg.Button("zapisz fragment"), sg.Button("mnozenie przez stala"),
                         sg.Button("kontrast"), sg.Button("gamma"), sg.Button("histogram"),
                         sg.Button("nieliniowy filtr medianowy"),
                         sg.Button("liniowy filtr usredniajacy"), sg.Button("sobel horizontal"),
                         sg.Button("sobel vertical"), sg.Button("laplacian")],
                        [sg.Button("zapisz"), sg.Button("resetuj obraz"), sg.Button("Wroc")]]
                       ]
        window = sg.Window("Aplikacja CPSiO", imageLayout)
        while True:
            event, values = window.read()
            if event == "Wroc" or event == sg.WIN_CLOSED:
                os.remove(defautlImage)
                break
            elif event == "resetuj obraz":
                image = cv2.imread(filename)
                cv2.imwrite(defautlImage, image)
                pass
            elif event == "mnozenie przez stala":
                image = multiply(image)
                pass
            elif event == "kontrast":
                image = contrast(image)
                pass
            elif event == "gamma":
                image = gamma(image)
                pass
            elif event == "histogram":
                histogram(image)
                pass
            elif event == "nieliniowy filtr medianowy":
                image = nonLinearMedianFilter(image)
                pass
            elif event == "liniowy filtr usredniajacy":
                image = linearAvgFilter(image)
                pass
            elif event == "szarosc":
                image = grey(image)
                pass
            elif event == "zapisz fragment":
                saveFragment(image)
                pass
            elif event == "zapisz":
                save(image)
                pass
            elif event == "sobel horizontal":
                sobelVertical(image)
                pass
            elif event == "sobel vertical":
                sobelHorizontal(image)
                pass
            elif event == "laplacian":
                laplacian(image)
                pass

            window["image"].update(filename=defautlImage)

    window.close()
    layout = [[sg.Text("""
            Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów
                    Kajetan Krasoń 252767, Kacper Małkowski 252724            """)],
              [sg.Button("Sygnały"), sg.Button("Obrazy"), sg.Button("Wyjdz")]]
    window = sg.Window("Aplikacja CPSiO", layout)

window.close()
if defautlImage:
    os.remove(defautlImage)
