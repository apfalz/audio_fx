import numpy            as np
import scipy.io.wavfile as wav
import librosa          as lib
import perlin           as per
import time
from os import listdir  as ls
import filters          as fil



from multiprocessing import Process, Queue


start_time = time.time()


def get_chunks(data, onsets, max_length=100000, min_length=11025, clean_chunks=True):
    output = []
    peaks  = []
    for o in range(len(onsets) -1):
        next = onsets[o+1]
        curr = onsets[o]
        diff = next - curr
        if diff > max_length:
            #truncate chunks that are too long
            next = curr + max_length
            output.append(data[curr:next])
            peaks.append(curr)
        if diff > min_length:
            if clean_chunks:
                output.append(clean_chunk(data[curr:next]))
            else:
                output.append(data[curr:next])
            peaks.append(curr)
    return peaks, output

def clean_chunk(chunk):

    ramp   = np.linspace(0.0, 1.0, len(chunk) // 20, endpoint=True)
    chunk  = np.array(chunk[:-len(chunk) // 10], copy=True)
    counter = len(chunk) - 1
    for i in range(len(ramp) - 1, -1, -1):
        chunk[counter] = chunk[counter] * ramp[i]
        counter -= 1
    return chunk



def mirror_chunks(chunks, onsets):
    output  = []
    new_ons = []
    for c in range(len(chunks)):
        output.append(np.hstack((chunks[c][::-1], chunks[c][1:])))
        new_ons.append(onsets[c] - len(chunks[c]))
    return new_ons, output


def get_surrounding_chunks(data, onsets, fs=44100, max_length=10, min_length=0.1):
    max_len  =     fs * max_length
    min_len  = int(fs * min_length)
    half_len = int(max_len / 2)
    #todo experiment with backtracking
    output   = []
    peaks    = []

    for o in range(1, len(onsets) - 1):
        left  = onsets[o-1]
        right = onsets[o+1]
        diff  = right - left
        if diff > max_len:
            new_left  = onsets[o] - half_len
            new_right = onsets[o] + half_len
            output.append(data[new_left:new_right])
            peaks.append(onsets[o])
        elif diff < min_len:
            new_left  = onsets[o] - min_len
            if new_left < 0:
                continue
            new_right = onsets[o] + min_len
            output.append(data[new_left:new_right])
            peaks.append(onsets[o])
        else:
            output.append(data[left:right])
            peaks.append(onsets[o])
    return peaks, output

def get_average_spacing(onsets):
    diffs = []
    for o in range(len(onsets) - 1):
        diffs.append(np.abs(onsets[o+1] - onsets[o]))#added abs() not sure if that's right....seems right....not sure.
    return int(np.mean(np.array(diffs)))

def apply_all_envelopes(data, env_len=0.25):
    output  = []
    for d in data:
        output.append(apply_envelope(d, env_len=0.25))
    print('finished applying all envelopes')
    return output

def apply_envelope(data, env_len=0.25):
    c      = np.copy(data)
    env    = int(len(c) * env_len)
    ramp   = np.linspace(0., 1., num=env)
    for i in range(env):
        c[i]            *= ramp[i]
        c[len(c)-(i+1)] *= ramp[i]
    return np.array(c)

def stretch_chunks(data, seed, scale_range=[0.125, 0.5]):
    np.random.seed(seed)
    range_val = scale_range[1] - scale_range[0]
    output = []
    for d in data:
        rand_val = (np.random.random() * range_val) + scale_range[0]
        output.append(lib.effects.time_stretch(d,rand_val))
    print('finished stretching chunks')
    return output

def proportional_stretch(chunks, strengths, seed, scale_range=[0.125, 0.5]):
    np.random.seed(seed)
    output = []
    for c in range(len(chunks)):
        amount = fil.scale(strengths[c], 0., 1., scale_range[0], scale_range[1]) + np.random.random()*scale_range[0]
        output.append(lib.effects.time_stretch(chunks[c], amount))
    return output

def place_chunks(chunks, peaks, target_len, seed):
    #this is final, all chunks flattened to single output
    np.random.seed(seed)
    avg    = get_average_spacing(peaks)
    output = np.zeros(target_len)
    for c in range(len(chunks)):
        offset = int(np.random.randint(avg) - (avg  / 2)) + peaks[c]
        length = len(chunks[c]) + offset
        if offset < 0 or length > target_len:
            continue
        try:
            output[offset:length] += chunks[c]
        except:
            if len(output) < length:
                output = np.hstack((output, np.zeros(length-len(output))))
                output[offset:length] += chunks[c]
            print('source: ' + str(output[offset:len(chunks[c])+offset].shape))
            print('target: ' + str(chunks[c].shape))
    print('finished placing chunks')
    return normalize(output)

def place_chunks_non_random(chunks):
    '''input is a dictionary of indices mapped to chunks'''
    output = np.zeros(1)
    print('num_chunks', len(chunks.keys()))
    for k in chunks.keys():
        chunk_len = len(chunks[k])
        # output[k:(chunk_len+k)] += chunks[k]
        error = False
        for s in range(len(chunks[k])):
            if len(output) < k + chunk_len:
                # if needed, extend output
                output = np.hstack((output, np.zeros(k+chunk_len)))
                print('newlen', len(output))
            try:
                output[k+s] += chunks[k][s]
            except:
                if error == False:
                    print('len(output)', len(output), 'len(output[k+s:])', len(output[k+s:]))
                    print('k+s', k+s)
                    output[k + s] += chunks[k][s]
                    error = True

    return normalize(output)

def two_channel_place_chunks(chunks):
    '''input is a dictionary of indices mapped to chunks'''
    output = [np.zeros(1), np.zeros(1)]
    counter = 0
    for k in chunks.keys():
        counter += 1
        chunk_len = len(chunks[k])
        error = False
        for s in range(len(chunks[k])):
            if len(output[counter%2]) < k + chunk_len:
                # if needed, extend output
                output[0] = np.hstack((output[0], np.zeros(k+chunk_len)))
                output[1] = np.hstack((output[1], np.zeros(k+chunk_len)))
            try:
                output[counter%2][k+s] += chunks[k][s]
            except:
                if error == False:
                    output[0] = np.hstack((output[0], np.zeros(k + chunk_len)))
                    output[1] = np.hstack((output[1], np.zeros(k + chunk_len)))
                    error = True
    return normalize(np.array(output).T)

def reverse_some(data, seed):
    np.random.seed(seed)
    output = []
    for d in data:
        if np.random.randint(3) == 1:
            output.append(d[::-1])
        else:
            output.append(d)
    return output

def pitch_shift_some(data, seed, fs=44100, vals=[12, -12]):
    np.random.seed(seed)
    output = []
    for d in data:
        rand_val = np.random.randint(4)
        if rand_val == 3:
            output.append(lib.effects.pitch_shift(d, fs, vals[0]))
        elif rand_val == 2:
            output.append(lib.effects.pitch_shift(d, fs, vals[1]))
        elif len(vals) == 3:
            output.append(lib.effects.pitch_shift(d, fs, vals[2]))
        else:
            output.append(d)
    return output

def reverse_and_pitch_shift_some(data, seed, fs=44100, vals=[12, -12]):
    np.random.seed(seed)
    output = []
    for d in data:
        shift_rand   = np.random.randint(3)
        reverse_rand = np.random.randint(2)
        if shift_rand == 2:
            if reverse_rand == 1:
                output.append(lib.effects.pitch_shift(d, fs, vals[0])[::-1])
            else:
                output.append(lib.effects.pitch_shift(d, fs, vals[0]))
        elif shift_rand == 0:
            if reverse_rand == 1:
                output.append(lib.effects.pitch_shift(d, fs, vals[1])[::-1])
            else:
                output.append(lib.effects.pitch_shift(d, fs, vals[1]))
        elif shift_rand == 1 and len(vals) == 3:
            if reverse_rand == 1:
                output.append(lib.effects.pitch_shift(d, fs, vals[2])[::-1])
            else:
                output.append(lib.effects.pitch_shift(d, fs, vals[2]))
        elif shift_rand == 1 and len(vals) != 3:
            if reverse_rand == 1:
                output.append(d[::-1])
            else:
                output.append(d)
        else:
            print('woopsie')

    return output

def gen_unique_fn(base, prefix):
    files     = ls('./outputs/')
    files     = [f for f in files if '.wav' in f]
    counter   = 0
    candidate = base + str(counter) + '.wav'
    done      = False
    while not done:
        if candidate in files:
            counter += 1
            candidate = base + str(counter) + '.wav'
        else:
            done = True
    candidate = prefix + candidate
    print('generated ' + candidate)
    return candidate

def normalize(data):
    return (data / max(np.amax(data), np.abs(np.amin(data)))) * 0.9

def mtof(pitch):
    pitch = float(pitch)
    return 440.0 * 2.0**((pitch - 69.0)/12.0)

def add_delays_to_chunk(data, seed, num_delays=3, max_offset=1.0, fs=44100):
    if type(data) == list:
        data = np.array(data)
    np.random.seed(seed)
    echos = []
    for i in range(num_delays):
        offset = int(np.random.random()*fs)
        echo   = (np.roll(data, offset) * 0.5 )
        echo   = fil.butter_lowpass_filter(echo, mtof(60.0), order=i+1)
        echos.append(echo + data)
    return normalize(np.sum(np.array(echos), axis=0))

def create_wiggly_pad(chunk, seed, scale_range=[0.25, 0.5]):
    np.random.seed(seed)
    output    = []
    range_val = scale_range[1] - scale_range[0]
    rand_val  = (np.random.random() * range_val) + scale_range[0]
    stretched = lib.effects.time_stretch(chunk, rand_val)
    low       = lib.effects.pitch_shift(stretched, fs, -12)
    high      = lib.effects.pitch_shift(stretched, fs,  12)
    low_perl  = per.perlin(1, len(low), 10, 1.0, 1.1, 0.5, normalize=True)
    mid_perl  = per.perlin(1, len(low), 10, 1.0, 1.1, 0.5, normalize=True)
    high_perl = per.perlin(1, len(low), 10, 1.0, 1.1, 0.5, normalize=True)
    output.append(low*low_perl)
    output.append(stretched*mid_perl)
    output.append(high*high_perl)
    output = np.sum(np.array(output), axis=0)
    return output

def wiggly_pads(chunks, seed):
    output = []
    for chunk in chunks:
        output.append(create_wiggly_pad(chunk, seed))
    return output

def get_strength_of_peaks(data, peaks, fs=44100, half_window=11025):
    # peaks     = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples')
    strengths = []
    for o in peaks:
        left     = max(o-half_window, 0)
        right    = min(o+half_window, len(data))
        vicinity = data[left:right]
        strengths.append(max(np.amax(vicinity), np.abs(np.amin(vicinity))))
    return peaks, strengths

def winnow_peaks(peaks, strengths, keep_percent=0.5):
    # peaks, strengths = get_strength_of_peaks(data, peaks, half_window=11025)
    #created sorted pairs of strengths and onset locations, keep only the strongest
    target = int(len(peaks) * keep_percent)
    pairs  = sorted(list(zip(strengths, peaks))) #sorted by strength
    output = []
    for i in range(len(pairs)-1, len(pairs)-1-target, -1):
        output.append(pairs[i])
    flipped = sorted([(i[1], i[0]) for i in output])#flip pairs and sort by onset, return re-flipped version
    final   = [(i[1], i[0]) for i in flipped]
    return final


def swoop(chunk, peak):
    '''
    get chunk, mirror it.
    get bell curve shaped intensity curve around peak.
    as intensity curve gets higher select more and more little chunks
    select chunks from where they occur in the area, but with small random offsets.
    possibly apply some swooshing of panning or filters, pitch shifting.
    '''
    mirrored  = lib.effects.time_stretch(np.hstack((chunk[::-1], chunk)), 0.1)

    ramp_len  = int(len(mirrored) / 2)
    intensity = np.hstack((np.linspace(1., 20., ramp_len), np.linspace(20., 1., ramp_len)))

    grain_len = int(44100) / 30
    rand_size = int(44100 / 30)
    grains    = {}
    #split the chunk into 10 parts
    seg_len   = int(len(mirrored)/10)


    #make a cloud with (sample from intensity) number of points at each seed
    cursor    = 0
    while cursor < len(mirrored):
        num_grains = int(intensity[int(cursor+(seg_len/2))%len(intensity)])
        for g in range(num_grains):
            try:
                start = np.random.randint(cursor, cursor+seg_len)
                end   = int(grain_len + np.random.randint(rand_size) + start)
                print('start', start)
                print('end', end, len(mirrored))
                raw = mirrored[start:end]
                raw = lib.effects.time_stretch(raw, 0.5)
                grain = apply_envelope(raw)
                grains[start] = grain
            except:
                pass
        cursor += seg_len
    output = place_chunks_non_random(grains)
    return output


def crossfade(source, target):
    #todo support other overlap amounts besides total length
    curve  = np.sqrt(0.5*(1.0+np.linspace(-1., 1., len(source), endpoint=True)))
    output = target*curve
    two    = (source * curve[::-1])
    return output + two

def do_crossfade(source, target, overlap_length):
    segments = []
    segments.append(source[:-overlap_length])
    fade     = crossfade(source[-overlap_length:], target[:overlap_length])
    segments.append(fade)
    segments.append(target[overlap_length:])
    return np.hstack(np.array(segments))

def fade_to_higher(chunk):
    lower = lib.effects.pitch_shift(chunk, 44100, -12)
    upper = lib.effects.pitch_shift(chunk, 44100, 24)
    return do_crossfade(lower, upper, len(lower))




#==========main methods========#
def stretch_and_reverse(chunks, peaks,  target_length, seed, fs=44100):
    #apply envelopes before
    chunks = stretch_chunks(chunks, seed)
    chunks = reverse_and_pitch_shift_some(chunks, seed)
    output = place_chunks(chunks, peaks, target_length, seed)
    fn     = gen_unique_fn('output_', 'outputs/')
    wav.write(fn, fs, output)

def mirrored_chunks(chunks, peaks, strengths, target_length, seed, fs=44100, shift_some=True):
    #apply envelopes ahead of time
    # chunks = stretch_chunks(chunks, seed) # this one is random
    chunks = proportional_stretch(chunks, strengths, seed, scale_range=[0.125, 0.5])
    if shift_some:
        chunks = pitch_shift_some(chunks, seed, fs=fs, vals=[12, -12])
    peaks, chunks = mirror_chunks(chunks, peaks)
    output = place_chunks(chunks, peaks, target_length, seed)
    fn     = gen_unique_fn('mirror_', 'outputs/')
    wav.write(fn, fs, output)

def tweeter(chunks, peaks, target_length, seed, fs=44100):
    #apply envelopes before
    chunks        = stretch_chunks(chunks, seed, scale_range=[6.0, 8.5])
    chunks        = reverse_and_pitch_shift_some(chunks, seed, vals=[24, 48, 24])
    output        = place_chunks(chunks, peaks, len(data), seed )
    output        = add_delays_to_chunk(output, seed)
    fn            = gen_unique_fn('output_', 'outputs/')
    print('fs:' + str(fs))
    wav.write(fn, fs, output)

def wiggler( chunks, peaks, target_len,  seed, fs=44100):
    #apply envelopes before
    chunks = wiggly_pads(chunks, seed)
    output = place_chunks(chunks, peaks, target_len, seed)
    fn     = gen_unique_fn('wiggler_', 'outputs/')
    wav.write(fn, fs, output)

def crickets(chunks,peaks, target_length, seed, fs=44100):
    #don't apply envelope before
    chunks = stretch_chunks(chunks, seed, scale_range=[6.0, 8.5])
    chunks = apply_all_envelopes(chunks, env_len=0.5)
    output = place_chunks(chunks, peaks, target_length, seed )
    output = add_delays_to_chunk(output, seed)
    fn     = gen_unique_fn('crickets_', 'outputs/')
    wav.write(fn, fs, output)



if __name__ == '__main__':
    input_fn = 'input/cuckoo.wav'
    fs, data = wav.read(input_fn)
    onsets   = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=True)
    #
    # peaks, chunks = get_chunks(data, onsets, min_length=10000)
    #
    # def make_whispy_cloud(chunk, onsets, seed):
    #     output = np.zeros()
    #
    # def get_whisps(chunk, onset, seed):
    #     np.random.seed(seed)
    #     length     = int(44100 * 0.1)
    #     num_whisps = 4
    #     whisps     = []
    #     for w in range(num_whisps):
    #         start = np.random.randint(len(chunk)-length)
    #         whisp = apply_envelope(chunk[start:start+length], env_len=0.5)
    #         whisp = fil.butter_lowpass_filter(whisp, 500.0, order=1)
    #         whisp = lib.effects.time_stretch(whisp, 0.025)
    #         whisp = lib.effects.pitch_shift(whisp, 44100, 12)
    #         whisps.append(whisp)
    #     base   = max([len(i) for i in whisps])
    #     output = np.zeros(base*2)
    #     output[:len(whisps[0])] = output[:len(whisps[0])+whisps[0]]
    #     return whisps
    #





    # peaks, chunks = get_chunks(data, onsets)
    peaks, strengths = get_strength_of_peaks(data, onsets, fs)
    pairs            = winnow_peaks(peaks, strengths, 0.5)
    strengths, peaks = zip(*pairs)
    peaks, chunks    = get_chunks(data, peaks)
    chunks           = apply_all_envelopes(chunks)

    procs = []
    for p in range(5):
        # procs.append(Process(target=crickets, args=(chunks, peaks,len(data), p, fs)))
        # procs.append(Process(target=tweeter, args=(chunks, peaks,len(data), p), kwargs={'fs': fs}))
        # procs.append(Process(target=wiggler, args=(chunks, peaks, len(data), p)))
        procs.append(Process(target=mirrored_chunks, args=(chunks, peaks, strengths, len(data), p, fs, False)))
        # procs.append(Process(target=stretch_and_reverse, args=(chunks, peaks, len(data), p, fs)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()



    print(time.time() - start_time)
    quit()
    # half     = lib.effects.time_stretch(data, 0.5)
