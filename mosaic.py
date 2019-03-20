import scipy.io.wavfile as wav
import librosa          as lib
import numpy            as np
import fx_functions        as pe
import filters          as fil
import warnings


def get_tile(data, target_len, slice_len=None, fs=44100):
    if slice_len == None:
        slice_len = target_len // 8
    left  = np.random.randint(0, len(data)-slice_len)
    right = left + slice_len
    tile  = data[left:right]
    scale = min(1.0, len(tile) / target_len)
    # print('before', len(tile))
    tile  = lib.effects.time_stretch(tile, scale)
    # print('len(tile)', len(tile))
    # print('scale', scale)

    ramp  = np.linspace(0., 1., 30)
    tile[   :30] *= ramp
    tile[-30:  ] *= ramp[::-1]
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

def envelope_mosaic(chunks, min_len, max_len, half_window=22050, fs=44100):
    cursor       = 0
    tiles        = []
    envelope     = pe.get_envelope(data, window_size=fs, order=5, normalize=True)
    wav.write('outputs/envelpope.wav', fs, envelope)
    quarter_len  = half_window // 2
    while cursor < len(data)-half_window:
        left     = max(0, cursor-fs)
        right    = min(len(data), cursor+fs)
        region   = data[left:right]
        strength = envelope[cursor+quarter_len]
        tile_len = fil.scale((1.0 - strength), 0., 1., min_len, max_len)
        rand_amt = np.random.randint(-tile_len//2, tile_len//2)
        tile_len += rand_amt
        tile     = get_tile(region, int(tile_len), 5500, fs)
        cursor  += len(tile)
        tiles.append(tile)
    return np.hstack(np.array(tiles))

if __name__ == '__main__':
    fs, data         = wav.read('input/mountain_man.wav')
    onsets           = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=False)
    peaks, strengths = pe.get_strength_of_peaks(data, onsets, fs)
    pairs            = pe.winnow_peaks(peaks, strengths, 0.5)
    strengths, peaks = zip(*pairs)
    peaks, chunks    = pe.get_chunks(data, peaks)
    chunks           = pe.apply_all_envelopes(chunks)

    output           = envelope_mosaic(data, 64, fs)
    fn = 'outputs/mosaic.wav'
    wav.write(fn, fs, output)
    print('done')
    quit()


