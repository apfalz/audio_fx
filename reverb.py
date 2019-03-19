import numpy             as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wav
import filters           as fil





def allpass(data, delay, a, b):
    output  = [0.0]*delay
    prepped = np.hstack([np.zeros(delay), data])
    for d in range(delay, len(data)):
        output.append((b*prepped[d]) + prepped[d-delay] + (a*output[d-delay]))
    return output


def reverb(data,  num):
    a       =  0.5
    b       = -0.5
    delay   = np.random.randint(100)
    output  = allpass(data, delay, a, b)
    cutoffs = np.linspace(22049.0, 1100.0, num-1)
    for n in range(num-1):
        delay += np.random.randint(100)
        if n % 2 == 1:
            a *= -1.0
            b *= -1.0
        output = allpass(output, delay + np.random.randint(100), a, b)
        output = fil.butter_highpass_filter(output, cutoffs[n], fs=44100, order=1)
    return np.array(output)





if __name__ == '__main__':
    fs, data = wav.read('input/sawbones.wav')

    filtered = fil.time_varying_biquad(data, np.linspace(100., 10000.0, len(data)+2), fs, 1.0 )
    # out = rev + data
    wav.write('outputs/hpf.wav', fs, filtered)
    print('done')

    quit()