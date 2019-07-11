import librosa          as lib
import numpy            as np
import scipy.io.wavfile as wav
import fx_functions     as fx


def separate_at_attacks(data, fs=44100):
    onsets = lib.onset.onset_detect(data, fs, units='samples')
    print(len(onsets))
    pairs = []
    for i in range(len(onsets)-1):
        pairs.append([onsets[i], onsets[i+1] -1])

    return pairs

def shuffle_pairs(pairs, half_window_size=2):
    shuffled = []
    for i in range(len(pairs)):
        # index = np.random.randint(max(0, i-half_window_size), min(i+half_window_size, len(pairs)))
        indices = list(range(max(0, i-half_window_size), min(len(pairs), i+half_window_size)))
        while len(indices) > 1:
            rand_index = np.random.randint(len(indices))
            real_index = indices[rand_index]
            shuffled.append(pairs[real_index])
            del indices[rand_index]
        shuffled.append(pairs[indices[0]])
    for j in shuffled:
        if (len(j)) != 2:
            print(j)
    return shuffled




def swap_shuffle(pairs, window_size=10, num_swaps=3):
    cursor = 0
    while cursor < len(pairs)-window_size:
        window = list(range(cursor, cursor+window_size))
        for i in range(num_swaps):
            rand_0 = window[np.random.randint(len(window))]
            rand_1 = window[np.random.randint(len(window))]
            pairs[rand_0], pairs[rand_1] = pairs[rand_1], pairs[rand_0]
        cursor += 1
    return pairs


def reassemble_shuffled_chunks(data, pairs, ramp_len=100):
    raw    = np.array(data, copy=True)
    ramp   = np.linspace(0., 1., ramp_len)
    output = []
    for i in pairs:
        print('pairs: ' + str(i))
        chunk = raw[i[0]:i[1]]
        for j in range(ramp_len):
            chunk[j]  *= ramp[j]
            chunk[-j] *= ramp[j]
        output.append(chunk)
    return np.hstack(np.array(output))


def jumble_up_chunks(data, fs=44100, ramp_len=1000):
    pairs = separate_at_attacks(data, fs=fs)
    pairs = swap_shuffle(pairs)
    output = reassemble_shuffled_chunks(data, pairs, ramp_len=ramp_len)
    return output

if __name__ == '__main__':
    fs, data = wav.read('input/just_guitar.wav')

    output = jumble_up_chunks(data)
    fn = fx.gen_unique_fn('jumble_up_chunks')
    wav.write(fn, fs, output)





    # quit()