from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav
import pre_echos        as pe
import numpy            as np


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff, fs=44100, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, midi_center_freq, fs, order=2):
    center_freq = pe.mtof(midi_center_freq)
    tolerance   = (pe.mtof(midi_center_freq+1.0) - center_freq) / 2.0
    print('midi', midi_center_freq)
    print('center', center_freq)
    print('tolerance', tolerance)
    b, a = butter_bandpass(center_freq - tolerance, center_freq + tolerance, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_with_octaves(data, midi_center_freq, fs, order=3):
    freqs  = [midi_center_freq, midi_center_freq+12.0, midi_center_freq-12.0]
    output = []
    for f in freqs:
        y = butter_bandpass_filter(data, f, fs, order=order)
        output.append(y)
    return output

if __name__ == '__main__':
    fs, data = wav.read('input/lil_deb.wav')

    output = bandpass_with_octaves(data, 63.0, fs)
    for o in range(len(output)):
        wav.write('bp' + str(o) + '.wav', fs, output[o])

    quit()