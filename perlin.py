import numpy             as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav







def create_layer(num_points, amp, target_len):
    points = np.random.random(int(num_points)) - 0.5
    interp = interp1d(np.linspace(0, num_points, num=num_points, endpoint=True), points, kind='cubic')
    return    (interp(np.linspace(0, num_points, num=target_len, endpoint=True)) * amp)[::-1]

def perlin(num_octaves, target_length, first_freq, first_amp, freq_scale, amp_scale, normalize=True):
    output   = []
    cur_freq = first_freq
    cur_amp  = first_amp
    for o in range(num_octaves):
        if o != 0:
            cur_freq *= freq_scale
            cur_amp  *= amp_scale
        output.append(create_layer(cur_freq, cur_amp, target_length))
    output = np.sum(np.array(output), axis=0)
    if normalize:
        return output / max(np.amax(output), np.abs(np.amin(output)))
    else:
        return output

def rms(data, window_size=512):
    vals = []
    cursor = 0
    while cursor < len(data):
        vals.append(np.sqrt(np.sum(np.square(data[cursor:cursor+window_size])) / float(window_size)))
        cursor += window_size
    interp = interp1d(np.arange(len(vals)), vals)
    return normalize(interp(np.linspace(0, len(vals)-1, num=len(data), endpoint=True)))

def time_varying_perlin(target_length, first_freq, freq_scale, amplitudes, num_octaves=3, normalize=True):
    output = []
    cur_freq = first_freq
    for o in range(num_octaves):
        if o != 0:
            cur_freq *= freq_scale
        output.append(create_layer(cur_freq, 1.0, target_length) * amplitudes[o])
    output = np.sum(np.array(output), axis=0)
    if normalize:
        return output / max(np.amax(output), np.abs(np.amin(output)))
    else:
        return output

def normalize(data):
    return data / max(np.amax(data), np.abs(np.amin(data)))

def create_amps(data):
    output = []
    output.append([1.0 - i for i in data])
    output.append([0.5]*len(data))
    output.append(data)
    return output

def scaled_amps(data):
    output = []

    output.append(data*0.25)
    output.append(data*0.5)
    output.append(data)
    return output




if __name__ == '__main__':
    data = perlin(3, 11025*3, 1000.0, 1.0, 2.0, 0.5, normalize=True)
    wav.write("outputs/perlin_noise.wav", 11025, data)
    print('done')
    quit()
