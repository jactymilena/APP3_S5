import math

import numpy as np
import matplotlib.pyplot as plt


def prob1_a_x1():
    n = 20
    t = np.arange(n)

    x1 = np.sin((0.1 * math.pi * t) + (math.pi/4))
    ft = np.fft.fft(x1)

    amp = np.abs(ft)
    ang = np.angle(ft)

    # Representation
    plt.stem(amp)
    plt.show()
    plt.stem(ang)
    plt.show()
    plt.stem(x1)
    plt.show()


def prob1_a_x2():
    N = 11
    arr = np.array([1, -1])

    x2 = np.tile(arr, N)
    ft = np.fft.fft(x2)

    amp = np.abs(ft)
    ang = np.angle(ft)

    # Representation
    plt.stem(x2)
    plt.show()
    plt.stem(amp)
    plt.show()
    plt.stem(ang)
    plt.show()


if __name__ == '__main__':
    print('allooo')
    prob1_a_x1()
    prob1_a_x2()
