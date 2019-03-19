import pre_echos         as pe
import numpy             as np
import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
from scipy import signal as sig
import shared_functions  as sf
import librosa           as lib
import filters           as fil




def bloop():
    pitches = [sf.mtof(60), sf.mtof(64), sf.mtof(66), sf.mtof(67), sf.mtof(69), sf.mtof(70), sf.mtof(72)]
    choice  = np.random.randint(len(pitches))
    length  = int(fs*0.05)
    return sf.osc(np.ones(length) * pitches[choice], sampling_rate=fs) * np.blackman(length)



def get_grain(chunk, target_len):
    #todo make scale more accurate
    seg_len = len(chunk) // 4
    index   = np.random.randint((len(chunk) - target_len))
    grain   = chunk[index:index+seg_len]
    scale   = len(grain) / target_len
    grain   = lib.effects.time_stretch(grain, scale)
    return grain

def get_compressed_chunk(data, midpoint, target_len, vicinity_size=44100*2):
    #choose random left index between midpoint-vicinity and midpoint
    left  = np.random.randint(max(0, midpoint-vicinity_size), midpoint)
    right = np.random.randint(midpoint, min(len(data), midpoint+vicinity_size))
    grain = data[left:right]
    scale = len(grain) / target_len
    grain = lib.effects.time_stretch(grain, scale)
    return grain

def get_points(scale=20):
    points = [0.15, 0.2, 0.339, 0.689, 0.960, 0.960, 0.689, 0.340, 0.201, 0.15]
    return [max(3, int(i*scale)) for i in points]

def get_midpoints(points, target_len=44100*8):
    #return indices of middle of points that will serve as reference points
    #to choose random chunks from their vicinities
    segment_len = target_len  // len(points)
    half_len    = segment_len // 2
    midpoints   = []
    for i in range(len(points)):
        midpoints.append(segment_len*i + half_len)
    return half_len, midpoints

def get_grains(points, midpoints, half_len, use_shift=False):
    chunk_dict = {}
    lens = np.linspace(44100//4, 44100//2, len(points))
    cutoffs = np.linspace(100., 22049.0, len(points))
    for m in range(len(points)):
        midpoint = midpoints[m]
        for p in range(points[m]):
            cur_point = np.random.randint(midpoint-half_len, midpoint+half_len)
            # grain     = get_grain(chunks[1], lens[m])
            grain     = get_compressed_chunk(data, midpoint, lens[m])
            if np.random.randint(len(points)) < m and use_shift:
                grain = lib.effects.pitch_shift(grain, fs, 12.0)
            grain = fil.butter_lowpass_filter(grain * np.blackman(len(grain)), cutoffs[m], fs=fs)
            chunk_dict[cur_point] = pe.normalize(grain)
    return chunk_dict





if __name__ == '__main__':
    fs, data = wav.read('input/mountain_man.wav')

    onsets              = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=True)
    peaks, chunks       = pe.get_chunks(data, onsets)

    points              = get_points(20)
    half_len, midpoints = get_midpoints(points)
    grains              = get_grains(points, midpoints, half_len)
    output              = pe.two_channel_place_chunks(grains)


    wav.write('outputs/grain_cloud.wav', fs, output)

    print('done')
    quit()













quit()


