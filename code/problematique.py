import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read


def fft_func(x, title='x[n]'):
    ft = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x))

    # Amplitude and angle
    amp = np.abs(ft)
    # ang = np.angle(ft)

    plt.plot(amp)
    plt.show()

    plt.plot(freq, amp)
    plt.xlim(0, 1)
    plt.show()

    # Plot x[n]
    # plt.title(f"{title} N={N}")
    # plt.stem(x)
    # plt.show()

    # Plot Amplitude et angle
    # fig, axis = plt.subplots(2)
    #
    # fig.suptitle(f"TFD({title}) N={N}")
    # axis[0].stem(amp)
    # axis[0].set(ylabel='Amplitude')
    # axis[1].stem(ang)
    # axis[1].set(ylabel='Phase')
    # plt.show()


def read_wave(filename):
    input_data = read(filename)
    return input_data[1]


if __name__ == '__main__':
    x = read_wave("../sounds/note_guitare_LAd.wav")

    plt.plot(x)
    plt.show()

    plt.stem(np.abs(x))
    plt.show()

    N = 32
    fft_func(x)

    print('allooo')
    print("Bonjour ! ")