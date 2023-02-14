import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write


def hamming_win(wave):
    win = np.hamming(len(wave)) 
    return win * wave


def get_freq_fondamenale(signal, freqs):
    amp = np.abs(signal)
    return freqs[np.argmax(amp)]


def find_nearest_idx(arr, val):
    arr = np.asarray(arr)
    return (np.abs(arr - val)).argmin()


def extract_parameters(signal, samplerate):
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
    
    return harmoniques, signal_amp, signal_phase


def find_rep_impul_order():
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


def create_env_temp(wave):
    N_h = find_rep_impul_order()
    coeff = np.multiply(np.ones(N_h), 1/N_h)
    conv = np.convolve(np.abs(wave), coeff)

    plt.plot(conv)
    plt.show()

    s = int((len(conv) - len(wave))/ 2) 

    # return conv[s:len(conv) - s - 1]
    return conv[0: len(wave)]


def create_octave_factors():
    notes_names = [ 'A', 'G#', 'G', 'F#', 'F', 'E', 'R#','R', 'C#', 'C', 'B' ]
    notes = {}
    for i, n in enumerate(notes_names, 1):
        notes[n] = -i

    notes[notes_names[len(notes_names)-1]] = 1

    return notes


def create_beethoven_masterpiece(notes_signals):
    note_len = int(len(notes_signals['G']) / 6)

    beethoven = notes_signals['G'][:note_len]
    beethoven = np.concatenate((beethoven, notes_signals['G'][:note_len]))
    beethoven = np.concatenate((beethoven, notes_signals['G'][:note_len]))
    beethoven = np.concatenate((beethoven, notes_signals['R#'][:note_len]))
    # manque silence
    beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    beethoven = np.concatenate((beethoven, notes_signals['F'][:note_len]))
    beethoven = np.concatenate((beethoven, notes_signals['R'][:note_len]))

    return beethoven


def create_signal(harmoniques, arr_amp, arr_phase, env_temp, s_len, fs):
    t = np.divide(np.arange(0, s_len), fs)
    y_sum = 0
    
    for i, f in enumerate(harmoniques):
        y_sum += arr_amp[i]*np.sin(f * 2 * np.pi * t + arr_phase[i]) 
    
    return np.multiply(y_sum, env_temp)
    # return y_sum


def create_note(signal, samplerate, filename):
    write(f"sounds/notes/{filename}.wav", samplerate, signal.astype(np.int32))


def create_beethoven_from_lad():
    # Lecture du fichier
    samplerate, data = read("sounds/note_guitare_LAd.wav")

    # Fenetre de hamming
    signal =  hamming_win(data)

    # Extract frequence, amplitude and phase
    harmoniques, signal_amp, signal_phase = extract_parameters(signal, samplerate)
    
    # Trouver l'enveloppe temporelle
    env_temp = create_env_temp(data)

    # Creation du signal
    lad = create_signal(harmoniques, signal_amp, signal_phase, env_temp, len(signal), samplerate) 
    
    create_note(lad, samplerate, 'A#')

    # Creation de toutes les notes
    notes = create_octave_factors()

    # notes_signals = {}
    # for k in notes:
    #     facteur = pow(2, notes[k] / 12)
    #     note_harmoniques = np.multiply(harmoniques, facteur)
    #     notes_signals[k] = create_signal(note_harmoniques, signal_amp, signal_phase, env_temp, len(signal), samplerate)

    # # Create beethoven
    # beethoven = create_beethoven_masterpiece(notes_signals)
    # create_note(beethoven, samplerate, 'beethoven')

def create_basson_low_pass(N, fe, plot=False):
    # Paramètres
    fc_0 = 1000
    fc_1 = 40
    n = np.arange(0 - (N / 2), (N / 2))
    m = (N * fc_1) / fe
    k = (2 * m) + 1

    # Normalisation des fréquences
    w_0 = (2 * np.pi * fc_0) / fe
    w_1 = (2 * np.pi * fc_1) / fe

    # Réponse impulsionnelle du filtre passe-bas
    h_lp = [1 / N * (np.sin(math.pi * n[i] * k / N) / np.sin(math.pi * n[i] / N)) if n[i] != 0 else k/N for i in range(N)]

    # Representation Graphique
    if plot:
        plt.title(f"Réponse impulsionnelle h[n] (N={N})")
        plt.plot(n, h_lp)
        plt.show()

    return h_lp, n, w_0 


def filter_basson_1000Hz(read_filename, write_filename):
    # Lecture du Basson
    # fe, data = read("sounds/note_basson_plus_sinus_1000_Hz.wav")
    fe, data = read(f"sounds/{read_filename}.wav")


    # Appliquer une fenêtre de Hamming
    basson_window = np.hamming(len(data)) * data 

    # Affichage
    plt.plot(basson_window)
    plt.show()

    N = 6000
    h_lp, n, w_0 = create_basson_low_pass(N, fe)

    # FFT du filtre passe-bas et transformation en dB
    hfft_pb = np.fft.fft(h_lp)
    h_db = 20 * np.log10(np.abs(hfft_pb)) 

    # Passe-bas -> Coupe-Bande
    h_cb = []
    for i in range(N):
        delta = 1 if(n[i] == 0) else 0
        h_cb.append(delta - np.multiply(2 * h_lp[i], np.cos(w_0 * n[i])))

    # fft du coupe bande pour afficher la frequence coupé
    cb_freq = np.fft.rfftfreq(len(h_cb), d=1./fe)
    hfft_cb = np.fft.rfft(h_cb)

    # Convolution du basson origine avec le filtre coupe-bande
    basson_synthese = np.convolve(basson_window, h_cb)

    # Création du nouveau basson
    write(f"sounds/{write_filename}.wav", fe, basson_synthese.astype(np.int16))

    # Affichage du basson de synthèse
    plt.title("Basson de synthèse")
    plt.plot(basson_synthese)
    plt.show() 

    # Graphique
    plt.title(f"Réponse impulsionnelle du Coupe-bande(N={N})")
    plt.plot(n,h_cb)
    plt.xlim([-5000, 1000])
    plt.show() 

    plt.title(f"Reponse en frequences du Coupe-bande(N={N})")
    plt.plot(cb_freq[:500], np.abs(hfft_cb[:500]))
    plt.show() 


if __name__ == '__main__':
    # create_beethoven_from_lad()
    filtered_filename = "basson-synthese"
    filter_basson_1000Hz("note_basson_plus_sinus_1000_Hz", "basson-synthese")
    filter_basson_1000Hz(filtered_filename, filtered_filename)

