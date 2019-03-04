import pre_echos         as pe
import numpy             as np
import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
from scipy import signal as sig
import shared_functions  as sf
import librosa           as lib
import filters           as fil

fs, data = wav.read('input/guitar.wav')


def bloop():
    pitches = [sf.mtof(60), sf.mtof(64), sf.mtof(66), sf.mtof(67), sf.mtof(69), sf.mtof(70), sf.mtof(72)]
    choice  = np.random.randint(len(pitches))
    length  = int(fs*0.05)
    return sf.osc(np.ones(length) * pitches[choice], sampling_rate=fs) * np.blackman(length)





# len_source_curve = 21
#
# source_curve = np.blackman(len_source_curve)
#
# points = []
# for i in range(len(source_curve)):
#     if i % 2 == 1:
#         points.append(source_curve[i])
#
# print(points)
# quit()

#get num chunks to create

def get_grain(chunk, target_len):
    #todo make scale more accurate
    seg_len = len(chunk) // 4
    index   = np.random.randint((len(chunk) - target_len))
    grain   = chunk[index:index+seg_len]
    scale   = len(grain) / target_len
    grain   = lib.effects.time_stretch(grain, scale)
    return grain


def get_points(scale=20):
    points = [0.009, 0.101, 0.339, 0.689, 0.960, 0.960, 0.689, 0.340, 0.101, 0.009]
    return [max(3, int(i*scale)) for i in points]

def get_midpoints(points, target_len=11025*8):
    segment_len = target_len  // len(points)
    half_len    = segment_len // 2
    midpoints   = []
    for i in range(len(points)):
        midpoints.append(segment_len*i + half_len)
    return half_len, midpoints

def get_grains(points, midpoints, half_len, chunks):
    chunk_dict = {}
    lens = np.linspace(11025//8, 11025//4, len(points))
    cutoffs = np.linspace(100., 1000.0, len(points))
    counter = 0
    for m in range(len(points)):
        for p in range(points[m]):
            midpoint  = midpoints[m]
            cur_point = np.random.randint(midpoint-half_len, midpoint+half_len)
            grain     = get_grain(chunks[1], lens[m])
            if np.random.randint(len(points)) < m:
                grain = lib.effects.pitch_shift(grain, fs, 12.0)
            grain = fil.butter_lowpass_filter(grain * np.blackman(len(grain)), cutoffs[m], fs=fs)
            chunk_dict[cur_point] = pe.normalize(grain)

            # chunk_dict[cur_point] = grain * np.blackman(len(grain))
    return chunk_dict






onsets = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=True)
peaks, chunks = pe.get_chunks(data, onsets)


points              = get_points(20)
half_len, midpoints = get_midpoints(points)
grains              = get_grains(points, midpoints, half_len, chunks)
output              = pe.two_channel_place_chunks(grains)
wav.write('outputs/grain_cloud.wav', fs, output)














quit()


