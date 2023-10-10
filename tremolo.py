import scipy.io.wavfile as wav
import fx_functions     as fx
import shared_functions as sf
import numpy            as np

def reactive_tremolo(data, min_freq=0.0, max_freq=10.0, slow_when_quiet=True):
    env = fx.get_envelope(data, normalize=True)
    if not slow_when_quiet:
        env = 1.0 - env
    env = (env*(max_freq-min_freq)) + min_freq
    mod = sf.osc(env)
    mod = (mod + 0.5) / 2.5
    return data * mod

def create_duck(ramp_len, min_val=0.1, zero_time=0):
    down = np.linspace(1.0, min_val, ramp_len)
    trough = np.ones(zero_time) * min_val
    if zero_time != 0:
        return np.hstack([down, trough, down[::-1]])
    else:
        return np.hstack([down, down[::-1]])


def reactive_tremolo_triangle_ducks(data, min_freq=0.0, max_freq=20.0, slow_when_quiet=True, fs=44100):
    env = fx.get_envelope(data, normalize=True)
    if not slow_when_quiet:
        env = 1.0 - env
    env = (env*(max_freq-min_freq)) + min_freq
    #pause between dips = fs/cur_freq
    dip    = create_duck(1000)
    output = []
    cursor = 0
    while cursor < len(data):
        cur_pause = int(fs / env[cursor])
        output.append(np.ones(cur_pause))
        output.append(dip)
        cursor += cur_pause
        cursor += len(dip)
    output = np.hstack(output)[:len(data)]
    wav.write('./outputs/delme.wav', 44100, output)
    return output * data





if __name__ == '__main__':
    fs, data = wav.read('input/lil_deb_44100fs.wav')

    output = reactive_tremolo_triangle_ducks(data, slow_when_quiet=False)
    fn     = fx.gen_unique_fn('tremolo_', 'outputs/')
    wav.write(fn, fs, output)
    print('done')
    quit()