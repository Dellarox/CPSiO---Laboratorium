import matplotlib as plt
import numpy as np
import PySimpleGUI as sg
from tkinter import Tk
from tkinter.filedialog import askopenfilename

layout = [[sg.Text("""
Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów
        Kajetan Krasoń 252767, Kacper Małkowski 252724            """)],
          [sg.Button("Wczytaj plik"), sg.Button("Wyjdz")]]


signal = [[sg.Button("Wroc")], [sg.Button("Wyswietl")]]
image = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("Wroc")], [sg.Button("Wczytaj plik")]]


window = sg.Window("Aplikacja CPSiO", layout)
while True:
    event, values = window.read()
    if event == "Wyjdz" or event == sg.WIN_CLOSED:
        break

    if event == "Wczytaj plik":
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        if filename != "":
            extension = filename.split(".")[1]
            if extension != "jpg" or extension != "png" or extension != "txt":
                sg.Popup("Nieprawidlowy format pliku")
            else:
                with open(filename, "r") as file:
                    data = file.read()
                if extension == "txt":
                    window = sg.Window("Aplikacja CPSiO", signal)
                elif extension == "png" or extension == "jpg":
                    window = sg.Window("Aplikacja CPSiO", image)
        else:
            sg.Popup("Nie wybrano pliku")

window.close()
