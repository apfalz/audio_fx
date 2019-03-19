import numpy as np
import scipy.io.wavfile as wav
import filters as fil
import librosa as lib
import pre_echos as pe
import perlin    as per


def evaporate(chunk, rate=0.25, use_perlin=False, fs=44100):
    chunk     = fil.time_varying_biquad(chunk, np.linspace(500., 10000., len(chunk) + 2), fs, 1.0, 'highpass')
    chunk     = lib.effects.hpss(chunk)[0]
    stretch   = lib.effects.time_stretch(chunk, rate)
    one       = lib.effects.pitch_shift(chunk, fs, 12)
    one       = lib.effects.time_stretch(one, rate)
    two       = lib.effects.pitch_shift(chunk, fs, 24)
    two       = lib.effects.time_stretch(two, rate)
    if use_perlin:
        stretch   = stretch * ((per.perlin(3, len(stretch), 50, 1.0, 1.1, 0.5, normalize=True) * 0.5)+0.5)
        one       = one * ((per.perlin(3, len(one), 50, 1.0, 1.1, 0.5, normalize=True)*0.5)+0.5)
        two       = two * ((per.perlin(3, len(two), 50, 1.0, 1.1, 0.5, normalize=True)*0.5)+0.5)


    curve     = 0.5 * np.tanh(np.linspace(-3.0, 3.0, len(one))) + 0.5
    stretch   = stretch * (curve[::-1])
    one       = one * np.blackman(len(two))
    two       = two * curve
    pad       = np.zeros(len(one) // 3)
    stretch = np.hstack([stretch, pad, pad])
    one = np.hstack([pad, one, pad])
    two = np.hstack([pad, pad, two])

    output    = one + two  + stretch
    return pe.normalize(output)

def evap_effect(chunks, peaks, fs=44100):
    evaped = {}
    for c in range(len(chunks)):
        evaped[peaks[c]] = evaporate(chunks[c], fs=fs)[:-len(chunks[c])//5]
    output = pe.place_chunks_non_random(evaped)
    return output

if __name__ == '__main__':
    fs, data      = wav.read('input/cuckoo.wav')
    data = data[:fs*10]
    onsets        = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=False)
    peaks, chunks = pe.get_chunks(data, onsets)
    chunks        = pe.apply_all_envelopes(chunks)

    # stretched = lib.effects.time_stretch(chunks[0], 0.25)
    # wav.write('outputs/s.wav', fs, stretched)
    # wav.write('outputs/o.wav', fs, chunks[0])
    # quit()



    o = evap_effect(chunks, peaks, fs=fs)
    wav.write('outputs/evap.wav', fs, o)


    print('done')
    quit()