import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import shared_functions as sf


def convolve(data, impulse):
    output = np.zeros(len(impulse))
    for i in range(len(data)):
        if i % 1000 == 0:
            print(i, end='\r')
        row    = data[i] * impulse
        output = np.append(output, 0.0)
        output[i:i+len(impulse)] += row
    return output / max(np.amax(output), np.abs(np.amin(output)))


def convolve_evolving_impulse(data, impulse_method, imp_len, imp_data):
    cur_imp = impulse_method(imp_data, 0, imp_len)
    output  = np.zeros(len(cur_imp))
    for i in range(len(data)):
        if i % 1000 == 0:
            print(i, end='\r')
        row    = data[i] * cur_imp
        output = np.append(output, 0.0)
        output[i:i+len(cur_imp)] += row
        cur_imp = impulse_method(data, i, imp_len)
    return output# / max(np.amax(output, np.abs(np.amin(output))))


fs, data = wav.read('input/lil_deb_44100fs.wav')

fs, imp_data = wav.read('input/cpe.wav')
data = data[:1000000]

imp_data = imp_data[:1000000]
# impulse = sf.osc([40.0]*(fs*2))
# impulse = data[fs:fs* 2]#[::-1]
# impulse = data[247306:247306+61827]#[::-1]
impulse = np.linspace(1.0, 0.0, 1000)

def noise_imp(length):
    return np.random.random(length)

def sliding_window(source, index, length):
    return source[index:index+length]

# conv = np.convolve(data, impulse)
# print('done with np conv')

convb = convolve_evolving_impulse(data, sliding_window, 1000, imp_data)
print('done with my conv')

# fn = 'outputs/np_convolution.wav'
# wav.write(fn, fs, conv)
# print('done')

fn = 'outputs/my_convolution.wav'
wav.write(fn, fs, convb)
print('done')

quit()