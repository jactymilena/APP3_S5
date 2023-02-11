import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def rep_impulsionnelle(fc, fe, N):
    n = np.arange(0 - (N / 2), (N / 2))
    m = N * (fc / fe)
    k = 2 * m + 1
    add_padding = False
    padding = 0 if add_padding is False else 50

    # Calcul reponse impulsionnelle
    h = [1 / N * (np.sin(math.pi * n[i] * k / N) / np.sin(math.pi * n[i] / N)) if n[i] != 0 else k/N for i in range(N)]
    h_ham = np.hamming(N) * h

    # fft, amplitude et phase
    hfft = np.fft.fft(h)

    h_db = [20 * np.log10(np.abs(x)) if x != 0 else 0 for x in hfft]
    w = [(i * 2 * math.pi) / (N + padding) for i in range(N + padding)]

    if add_padding is True:
        h_db.extend([0] * padding)

    # Representation
    plt.title(f"Réponse impulsionnelle h[n] (N={N})")
    plt.stem(n, h)
    plt.show()

    plt.title(f"Réponse impulsionnelle h_ham[n] (N={N})")
    plt.stem(n, h_ham)
    plt.show()

    plt.title(f"Amplitude (N={N})")
    plt.xlim(0, math.pi)
    plt.plot(w, h_db)
    plt.show()

def prob1():
    plt.close()
    fc = 2000
    fe = 16000
    N = [16, 32, 64]

    # rep_impulsionnelle(fc, fe, N[0])
    # rep_impulsionnelle(fc, fe, N[1])
    rep_impulsionnelle(fc, fe, N[2])


if __name__ == '__main__':
    prob1()
