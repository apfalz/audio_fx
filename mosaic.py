import scipy.io.wavfile as wav
import librosa          as lib
import numpy            as np
import pre_echos        as pe
import filters          as fil
import warnings


def get_tile(data, target_len, slice_len=None, fs=44100):
    if slice_len == None:
        slice_len = target_len // 8
    left  = np.random.randint(0, len(data)-slice_len)
    right = left + target_len
    tile  = data[left:right]
    scale = len(tile) / target_len
    tile = lib.effects.time_stretch(tile, scale)
    ramp = np.linspace(0., 1., 30)
    tile[   :30] *= ramp
    tile[-30:  ] *= ramp[::-1]
    try:
        tile = fil.biquad(tile, np.random.randint(fs//2), fs, np.random.randint(40), 'highpass')
    except Warning:
        print('filter overflowed')
    return tile

def fill_chunk_with_tiles(data, tile_len, slice_len, seed,  rand=True, fs=44100):
    # np.random.seed(seed)
    tiles  = []
    length = 0
    cur_tile_len = tile_len
    while length < len(data):
        if rand:
            cur_tile_len = tile_len + np.random.randint(-tile_len//2, tile_len//2)
        tile = get_tile(data, cur_tile_len, slice_len)
        tiles.append(tile)
        length += len(tile)
    return np.hstack(np.array(tiles))

def mosaic_effect(chunks, tile_len, slice_len, seed, num_layers=2, rand=True, fs=44100):
    layers = []
    for l in range(num_layers):
        layer = []
        for c in chunks:
            tiled_chunk = fill_chunk_with_tiles(c, tile_len, slice_len, seed, rand=rand)
            layer.append(tiled_chunk)
        layer = np.hstack(np.array(layer))
        layers.append(layer)
    #figure out which is the longest, for padding the shorter ones.
    longest = len(layers[0])
    for l in layers:
        if len(l) > longest:
            longest = len(l)
    for l in range(len(layers)):
        if len(layers[l]) < longest:
            layers[l] = np.hstack([layers[l], np.zeros(longest-len(layers[l]))])
    return np.sum(np.array(layers), axis=0)



if __name__ == '__main__':
    fs, data         = wav.read('input/cuckoo.wav')
    onsets           = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=False)
    peaks, strengths = pe.get_strength_of_peaks(data, onsets, fs)
    pairs            = pe.winnow_peaks(peaks, strengths, 0.5)
    strengths, peaks = zip(*pairs)
    peaks, chunks    = pe.get_chunks(data, peaks)
    chunks           = pe.apply_all_envelopes(chunks)

    output           = mosaic_effect(chunks, fs//2, fs//64, 1, 1)

    fn = 'outputs/mosaic.wav'
    wav.write(fn, fs, output)
    print('done')
    quit()


