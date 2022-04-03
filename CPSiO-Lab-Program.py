import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import imageio as iio

signal = np.loadtxt('dane\ekg1.txt')
image = iio.imread('wykresy\ekg1.png')

layout = [[sg.Text("""
Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów
        Kajetan Krasoń 252767, Kacper Małkowski 252724            """)],
          [sg.Button("Wczytaj plik"), sg.Button("Wyjdz")]]
window = sg.Window("Aplikacja CPSiO", layout)


def showPlot(data, xlabel=None, ylabel=None):
    plt.clf()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def signal1(frequency):
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
            if values["t1"] == "":
                t1 = 0
            else:
                t1 = float(values["t1"]) * frequency

            if values["t2"] == "":
                t2 = signal.size
            else:
                t2 = float(values["t2"]) * frequency

            showPlot(signal[int(t1):int(t2)], "Probki", " ")
    showSignal.close()


def signal2():
    series= np.sin(2 * np.pi * 50 * np.arange(0, 65536) / 65536)
    # discreet fourier transform
    fourier = np.fft.fft(series)
    print (fourier)
    # amplitude spectrum
    amplitude = np.abs(fourier)
    print  (amplitude)
    # frequency spectrum
    frequency = np.fft.fftfreq(65536, d=1)
    print (frequency)

    showPlot([fourier, amplitude,frequency], "", "")
    pass


def signal3():
    pass


def signal4():
    pass


while True:
    event, values = window.read()
    if event == "Wyjdz" or event == sg.WIN_CLOSED:
        break

    if event == "Wczytaj plik":
        filename = askopenfilename()
        if filename != "":
            extension = filename.split(".")[1]
            if extension != 'jpg' and extension != 'png' and extension != 'txt':
                sg.Popup("Nieprawidlowy format pliku")
            else:
                if extension == "txt":
                    signal = np.loadtxt(filename)
                    window.close()
                    if filename.find("ekg1.txt"):
                        f = 1000
                    elif filename.find("ekg100.txt") or filename.find("ekg_noise.txt"):
                        f = 360
                    signalLayout = [[sg.Button("cw.1"), sg.Button("cw.2"), sg.Button("cw.3"), sg.Button("cw.4")],
                                    [sg.Button("Wroc")]]
                    window = sg.Window("Aplikacja CPSiO", signalLayout)
                    while True:
                        event, values = window.read()
                        if event == "Wroc" or event == sg.WIN_CLOSED:
                            break
                        if event == "cw.1":
                            signal1(f)
                        if event == "cw.2":
                            signal2()
                        if event == "cw.3":
                            signal3()
                        if event == "cw.4":
                            signal4()
                elif extension == "png" or extension == "jpg":
                    image = iio.imread(filename)
                    window.close()
                    imageLayout = [[sg.Button("Wroc")], [sg.Button("Wyswietl")]]
                    window = sg.Window("Aplikacja CPSiO", imageLayout)
                    while True:
                        event, values = window.read()
                        if event == "Wroc" or event == sg.WIN_CLOSED:
                            break
                        if event == "Wyswietl":
                            sg.Popup(image)
            window.close()
            layout = [[sg.Text("""
            Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów
                    Kajetan Krasoń 252767, Kacper Małkowski 252724            """)],
                      [sg.Button("Wczytaj plik"), sg.Button("Wyjdz")]]
            window = sg.Window("Aplikacja CPSiO", layout)
        else:
            sg.Popup("Nie wybrano pliku")

window.close()
