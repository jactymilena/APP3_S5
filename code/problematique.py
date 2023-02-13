import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
import scipy 


def hamming_win(wave):
    win = np.hamming(len(wave)) 
    return win * wave


def get_freq_fondamenale(signal, freqs):
    amp = np.abs(signal)
    return freqs[np.argmax(amp)]


def find_nearest_idx(arr, val):
    arr = np.asarray(arr)
    return (np.abs(arr - val)).argmin()


def find_rep_impul_N():
    H_3 = pow(10, -3/20)
    w_h = np.pi / 1000

    # print(f"On veut {H_3}")

    # N_h entre 800 et 890
    sum_rep = 0
    for N_h in range(880, 890):
        for n in range(N_h):
            sum_rep += np.exp(-1j * w_h * n)

        sum_rep *= 1/N_h
    # print(f"N = {N_h} rep : {np.abs(sum_rep)}")
    # Le N a ete trouve en comparant visuellement : 886
    return 886


def create_env_temp(signal):
    N_h = find_rep_impul_N()
    coeff = np.multiply(np.ones(N_h), 1/N_h)
    conv = np.convolve(np.abs(signal), coeff)

    plt.plot(conv)
    plt.show()

    s = int((len(conv) - len(signal))/ 2) 
    print(s)

    # return conv[s:len(conv) - s - 1]
    return conv[0: len(signal)]

def create_signal(harmoniques, arr_amp, arr_phase, env_temp):
    t = np.divide(np.arange(0, len(signal)), samplerate)
    y_sum = 0
    
    for i, f in enumerate(harmoniques):
        y_sum += arr_amp[i]*np.sin(f * 2 * np.pi * t + arr_phase[i]) 

    # plt.plot(y_sum)
    # plt.show()

    # test = np.multiply(y_sum, env_temp)
    # plt.plot(test)
    # plt.show()
    
    return np.multiply(y_sum, env_temp)
    # return y_sum

def create_note(signal, samplerate, filename):
    write(f"sounds/notes/{filename}.wav", samplerate, signal.astype(np.int32))


if __name__ == '__main__':
    # Lecture du fichier
    samplerate, data = read("sounds/note_guitare_LAd.wav")

    # Fenetre de hamming
    signal =  hamming_win(data)

    # FFT
    fft_spect = np.fft.rfft(signal)
    amp_fft_spect = np.abs(fft_spect)
    phase_fft_spect = np.angle(fft_spect)   
    fft_freq = np.fft.rfftfreq(signal.size, d=1./samplerate)

    # Reherche de la frequence fondamentale
    N = 32
    freq_fondamentale = get_freq_fondamenale(fft_spect, fft_freq)
    tmp_harmoniques = np.arange(freq_fondamentale, (N + 1) * freq_fondamentale, freq_fondamentale)

    # Trouver la frequence, l'amplitude et la phase des N harmoniques
    harmoniques = []
    signal_amp = []
    signal_phase = []

    for f_h in tmp_harmoniques:
        idx = find_nearest_idx(fft_freq, f_h)
        harmoniques.append(fft_freq[idx])
        signal_amp.append(amp_fft_spect[idx])
        signal_phase.append(phase_fft_spect[idx])
    
    # Trouver l'enveloppe temporelle
    # Trouver la valeur du N de la reponse impulsionnelle 
    env_temp = create_env_temp(data)
    plt.plot(env_temp)
    plt.show()

    # Creation du signal
    new_signal = create_signal(harmoniques, signal_amp, signal_phase, env_temp) 
    
    create_note(new_signal, samplerate, 'A#')

    # Creation de toutes les notes
    notes_names = [ 'A', 'G#', 'G', 'F#', 'F', 'E', 'R#','R', 'C#', 'C', 'B' ]
    notes = {}
    for i, n in enumerate(notes_names, 1):
        notes[n] = -i

    notes[notes_names[len(notes_names)-1]] = 1

    # notes_signals = {}
    # for k in notes:
    #     facteur = pow(2, notes[k] / 12)
    #     note_harmoniques = np.multiply(harmoniques, facteur)
    #     notes_signals[k] = create_signal(note_harmoniques, signal_amp, signal_phase, env_temp)

    # # Create beethoven
    # note_len = int(len(notes_signals['G']) / 6)
    # beethoven = notes_signals['G'][:note_len]
    # beethoven = np.concatenate((beethoven, notes_signals['G'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['G'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['R#'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    # beethoven = np.concatenate((beethoven, notes_signals['R'][:note_len]))

    # create_note(beethoven, samplerate, 'beethoven')


    # plt.plot(env_temp)
    # plt.show()

    # plt.stem(harmoniques, signal_phase)
    # plt.title('Phase')
    # plt.show()

    # plt.stem(harmoniques,  signal_amp)
    # plt.title('Amplitude')
    # plt.yscale('log')
    # plt.show()

    # plt.plot(fft_freq, np.abs(fft_spect))
    # plt.xlim(0, 20000)
    # plt.show()


    # plt.plot(signal)
    # plt.show()

    # plt.plot(fenetre)
    # plt.show()

