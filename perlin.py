import numpy             as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav







def create_layer(num_points, amp, target_len):
    points = np.random.random(int(num_points)) - 0.5
    interp = interp1d(np.linspace(0, num_points, num=num_points, endpoint=True), points, kind='cubic')
    return     interp(np.linspace(0, num_points, num=target_len, endpoint=True)) * amp

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





if __name__ == '__main__':

    d = perlin(5,  44100*3, 500, 1.0, 2.0, 0.9  )
    fn = 'perlin.wav'
    wav.write(fn, 44100, d)
    print('saved ' + fn)
    # plt.plot(d)
    # plt.show()
