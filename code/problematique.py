import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read


def fft_func(signal, fs):
    N = 32
    fft_spect = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.size, d=1./fs)

    # Amplitude and angle
    amp_fft_spect = np.abs(fft_spect)
    # ang = np.angle(ft)

    plt.plot(amp_fft_spect)
    plt.show()

    plt.plot(freq, amp_fft_spect)
    plt.xlim(0, 10000)
    plt.show()

    ind = np.argpartition(amp_fft_spect, -N)[-N:]
    for i in ind:
        print(f"freq : {freq[i]}, amp : {amp_fft_spect[i]}")




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
    samplerate, input_data = read(filename)
    return samplerate, input_data


if __name__ == '__main__':
    samplerate, x = read_wave("../sounds/note_guitare_LAd.wav")

    plt.plot(x)
    plt.show()

    plt.stem(np.abs(x))
    plt.show()

    N = 32
    fft_func(x, samplerate)

    print('allooo')
    print("Bonjour ! ")