import numpy            as np
import fx_functions        as pe
import scipy.io.wavfile as wav
import librosa          as lib
from multiprocessing import Process
import shared_functions as sf



if __name__ == '__main__':
    input_fn = 'input/lil_deb_44100fs.wav'
    fs, data = wav.read(input_fn)

    env      = pe.get_envelope(data, normalize=True)

    lo_freq  = 10.0
    hi_freq  = 40.0

    #change amplitude
    env      = env * hi_freq - lo_freq

    #add offset
    env      = env + lo_freq

    mod      = sf.osc(env, sampling_rate=fs)

    output   = data * mod
    fn       = pe.gen_unique_fn('ring_mod_', 'outputs/')
    wav.write(fn, fs, output)
    quit()