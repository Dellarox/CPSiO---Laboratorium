import numpy as np
import matplotlib.pyplot as plt
import os

print("""
--------------------------------------------------------------------
|Laboratorium z przedmiotu Cyfrowe Przetwarzanie Sygnałów i Obrazów|
|        Kajetan Krasoń 252767, Kacper Małkowski 252724            |
--------------------------------------------------------------------
Wybierz operację, którą chcesz wykonać:
1. .
2. .
3. .
""")

#input dolnaGranica and gornaGranica
#dolnaGranica = int(input("Podaj dolna granicę przedziału: "))
#gornaGranica = int(input("Podaj górną granicę przedziału: "))
dolnaGranica = 1000
gornaGranica = 2000

#make gui where user can write doldnaGranica and gornaGranica

#read data from txt file
dataEKG1 = np.loadtxt('dane\ekg1.txt')

#make plot for ekg1.txt in interval [dolnaGranica, gornaGranica] and save it to wykresy folder as ekg1.png
plt.plot(dataEKG1[dolnaGranica:gornaGranica])
plt.title('EKG1')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda [Hz]')
plt.grid(True)
plt.savefig('wykresy\ekg1.png')
plt.show()

#generate a sample sequence corresponding to a 50 Hz sine wave amd a length of 65536
sampleSequence = np.sin(2 * np.pi * 50 * np.arange(0, 65536) / 65536)

#generate a fft of the sampleSequence and show its magnitude spectrum on plot in frequency  (0, 65536/2)
fftSampleSequence = np.fft.fft(sampleSequence)
plt.plot(np.abs(fftSampleSequence[0:65536//2]))
plt.title('Magnitude spectrum of the sample sequence')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

#generate a sample sequence mixture of two sine waves with frequencies of 50 and 60 Hz
sampleSequenceMixture = np.sin(2 * np.pi * 50 * np.arange(0, 65536) / 65536) + np.sin(2 * np.pi * 60 * np.arange(0, 65536) / 65536)

#generate a fft of the sampleSequenceMixture and show its magnitude spectrum on plot in frequency  (0, 65536/2)
fftSampleSequenceMixture = np.fft.fft(sampleSequenceMixture)
plt.plot(np.abs(fftSampleSequenceMixture[0:65536//2]))
plt.title('Magnitude spectrum of the sample sequence mixture')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

#generate a sample sequence corresponding to a 50 Hz sine wave amd a length of 32768
sampleSequence = np.sin(2 * np.pi * 50 * np.arange(0, 32768) / 32768)

#generate a fft of the sampleSequence and show its magnitude spectrum on plot in frequency  (0, 32768/2)
fftSampleSequence = np.fft.fft(sampleSequence)
plt.plot(np.abs(fftSampleSequence[0:32768//2]))
plt.title('Magnitude spectrum of the sample sequence')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
