from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav
import fx_functions        as pe
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

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
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


def get_biquad_coeffs(cf, fs, Q, mode):
    #q [0.1, 10.0]
    k = np.tan(np.pi * cf / fs)
    denom = Q + (k * k)
    norm = 1.0 / ((1.0 + k) / denom)
    if mode == 'highpass':
        a = norm
        b = -2.0 * norm
        c = norm
        d =  2.0 * (k*k - 1.0) * norm
        e = (1.0 - k / denom) * norm
    elif mode == 'lowpass':
        a = k * k * norm
        b = 2.0 * a
        c = a
        d = 2.0 * (k * k - 1.0) * norm
        e = (1.0 - k / denom) * norm
    return a, b, c, d, e, k, norm

def biquad(data, cf, fs, Q, mode):
    a, b, c, d, e, k, norm = get_biquad_coeffs(cf, fs, Q, mode)
    output = [0., 0.]
    data   = np.hstack([np.zeros(2), data])
    big    = 1.0#np.finfo('d').max
    small  = -1.0#np.finfo('d').min
    for i in range(2, len(data)):
        o_0 = min(output[i-1], big)
        o_0 = max(output[i-1], small)
        o_1 = min(output[i-2], big)
        o_1 = max(output[i-2], small)
        output.append((a*data[i] + (b*data[i-1])  + (c*data[i-2]) - (d*o_0) - (e*o_1)))
    return np.array(output)

def time_varying_biquad(data, cf, fs, Q, mode):
    output = [0., 0.,]
    data   = np.hstack([np.zeros(2), data])
    for i in range(2, len(data)):
        a, b, c, d, e, k, norm = get_biquad_coeffs(cf[i], fs, Q, mode)
        output.append((a*data[i] + (b*data[i-1])  + (c*data[i-2]) - (d*output[i-1]) - (e*output[i-2])))
    return np.array(output[2:])

def time_varying_biquad_2(data, cf, fs, Q):
    output = [0., 0.]
    v      = [0., 0.]
    data   = np.hstack([np.zeros(2), data])
    for i in range(2, len(data)):
        a, b, c, d, e, k, norm = get_biquad_coeffs(cf[i], fs, Q)
        v.append(data[i] - (a*v[i-1]) - (b*v[i-2]))
        output.append((c*v[i]) + (d*v[i-1]) + (e*v[i-2]))
    return np.array(output)


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