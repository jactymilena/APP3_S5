import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def rep_impulsionnelle(fc, fe, N):
    n = np.arange(0 - (N / 2), (N / 2))
    print('n : ' + str(n))

    m = N * (fc / fe)
    k = 2 * m + 1
    print('m : ' + str(m))
    print('k : ' + str(k))

    h = []
    # h_f = []
    h_ham = []

    for i in range(N):
        if n[i] != 0:
            h.append((1 / N) * (np.sin(math.pi * n[i] * k / N) / np.sin(math.pi * n[i] / N)))
        else:
            h.append(k / N)
        # h_f.append(20 * math.log(h[i]))

    h_ham = np.hamming(N) * h
    print(h)

    #
    # h.append(0)
    # h.append(0)
    # h.append(0)

    # hfft = np.fft.fft(h)

    plt.title(f"Réponse impulsionnelle h[n] (N={N})")
    plt.stem(n, h)
    plt.show()

    plt.title(f"Réponse impulsionnelle h_ham[n] (N={N})")
    plt.stem(n, h_ham)
    plt.show()

    # amp = np.abs(hfft)
    # ang = np.angle(hfft)
    #
    # plt.title(f"Amplitude (N={N})")
    # plt.stem(amp)
    # plt.show()
    #
    # plt.title(f"Phase (N={N})")
    # plt.stem(ang)
    # plt.show()


def prob1():
    fc = 2000
    fe = 16000
    N = [16, 32, 64]

    rep_impulsionnelle(fc, fe, N[0])
    # rep_impulsionnelle(fc, fe, N[1])
    # rep_impulsionnelle(fc, fe, N[2])


if __name__ == '__main__':
    prob1()
