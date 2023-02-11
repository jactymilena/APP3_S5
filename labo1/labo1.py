import math

import numpy as np
import matplotlib.pyplot as plt


def fft_func(x, title='x[n]'):
    # fft
    ft = np.fft.fft(x)

    # Amplitude and angle
    amp = np.abs(ft)
    ang = np.angle(ft)

    # Plot x[n]
    plt.title(title)
    plt.stem(x)
    plt.show()

    # Plot Amplitude et angle
    fig, axis = plt.subplots(2)

    fig.suptitle(f"TFD({title})")
    axis[0].stem(amp)
    axis[0].set(ylabel='Amplitude')
    axis[1].stem(ang)
    axis[1].set(ylabel='Angle')
    plt.show()


def prob1_a_x1():
    N = 22
    t = np.arange(N)
    x1 = np.sin((0.1 * math.pi * t) + (math.pi/4))
    fft_func(x1, 'x_1[n]')


def prob1_a_x2():
    N = 11
    arr = np.array([1, -1])
    x2 = np.tile(arr, N)
    fft_func(x2, 'x_2[n]')


if __name__ == '__main__':
    print('Laboratoire 1')
    prob1_a_x1()
    prob1_a_x2()
