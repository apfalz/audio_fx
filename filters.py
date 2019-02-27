from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav
import pre_echos        as pe
import numpy            as np
import shared_functions as sf


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
    b, a = butter_bandpass(center_freq - tolerance, center_freq + tolerance, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_with_octaves(data, midi_center_freq, fs, order=3):
    freqs  = [midi_center_freq, midi_center_freq+12.0, midi_center_freq-12.0]
    output = np.zeros(len(data))
    for f in freqs:
        y = butter_bandpass_filter(data, f, fs, order=order)
        output = output + y
    return output

def get_amplitude_signal(data, midi_center_freq, fs, min_ramp=0.1, max_ramp=1, order=3):
    filtered          = bandpass_with_octaves(data, midi_center_freq, fs, order=order)
    pairs             = pe.winnow_peaks(filtered, keep_percent=0.1)
    strengths         = [i[0] for i in pairs]
    onsets            = [i[1] for i in pairs]
    output            = np.zeros(len(data))
    if max(strengths) > 1.0:
        strengths     = np.array(strengths) / max(strengths)
    min_ramp          = int(fs*min_ramp)
    max_ramp          = int(fs*max_ramp)
    for i in range(len(strengths)):
        ramp = np.linspace(0., strengths[i], scale(strengths[i], 0., 1., min_ramp, max_ramp))
        if onsets[i] < len(ramp):
            ramp_in = np.linspace(0., 1., onsets[i])
            output[:len(ramp_in)] += ramp_in
            output[onsets[i]:onsets[i]+len(ramp)] += ramp[::-1]
        else:
            output[onsets[i]-len(ramp):onsets[i]            ] += ramp
            output[onsets[i]          :onsets[i] + len(ramp)] += ramp[::-1]
    return output

def scale(value, in_lo, in_hi, out_lo, out_hi):
    out_range  = out_hi - out_lo
    in_range   = in_hi  - in_lo
    normalized = (value - in_lo) / in_range
    return (normalized * out_range) + out_lo


if __name__ == '__main__':
    fs, data = wav.read('input/lil_deb.wav')
    pitches  = [65.0, 68.0, 61.0, 60.0, 63.0]
    for p in range(len(pitches)):
        env    = get_amplitude_signal(data, pitches[p], fs)
        o      = sf.osc([sf.mtof(pitches[p])]*len(env)) * env
        wav.write('test'+str(p)+'.wav', fs, pe.normalize(o))
    quit()
    # quit()
    # quit()
    # for o in range(len(output)):
    #     wav.write('bp' + str(o) + '.wav', fs, output[o])
    # quit()