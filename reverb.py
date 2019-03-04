import numpy             as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wav
import filters           as fil



def allpass_b(data, delay=10):
    output  = []
    #prepend zeros
    data    = np.hstack([np.zeros(delay), data])
    stored  = data[:delay]

    for d in range(delay, len(data)):
        a = data[d] + (stored[-1] * 0.5)
        output.append(a * -0.5 + stored[-1])
        stored = np.roll(stored, 1)
        stored[0] = a
    return output

def allpass_a(data, delay=10):
    output  = []
    #prepend zeros
    data    = np.hstack([np.zeros(delay), data])
    stored  = data[:delay]

    for d in range(delay, len(data)):
        a = data[d] + (stored[-1] * -0.5)
        output.append(a * 0.5 + stored[-1])
        stored = np.roll(stored, 1)
        stored[0] = a
    return output

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
    cutoffs = np.linspace(22049.0, 100.0, num-1)
    for n in range(num-1):
        delay += np.random.randint(100)
        if n % 2 == 1:
            a *= -1.0
            b *= -1.0
        output = allpass(output, np.random.randint(1024), a, b)
        output = fil.butter_lowpass_filter(output, cutoffs[n], fs=44100, order=1)
    return np.array(output)







def simple_delay():
    output  = []
    delay   = 10
    data    = np.zeros(100)
    data[0] = 1
    data    = np.hstack([np.zeros(delay), data])
    stored  = data[0:delay]
    for i in range(delay, len(data)):
        sample = data[i] + (stored[-1] * 0.9)
        output.append(sample)
        stored = np.roll(stored, 1)
        stored[0] = sample
    return output



if __name__ == '__main__':
    fs, data = wav.read('input/sawbones.wav')

    rev = reverb(data, 30)
    # out = rev + data
    wav.write('outputs/reverb.wav', fs, rev)
    print('done')

    quit()