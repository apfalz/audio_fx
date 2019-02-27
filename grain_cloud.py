import pre_echos         as pe
import numpy             as np
import scipy.io.wavfile  as wav
import matplotlib.pyplot as plt
from scipy import signal as sig
import shared_functions  as sf


fs = 11025

def bloop():
    pitches = [sf.mtof(60), sf.mtof(64), sf.mtof(66), sf.mtof(67), sf.mtof(69), sf.mtof(70), sf.mtof(72)]
    choice  = np.random.randint(len(pitches))
    length  = int(fs*0. e5)
    return sf.osc(np.ones(length) * pitches[choice], sampling_rate=fs) * np.blackman(length)

len_source_curve = 21

source_curve = np.blackman(len_source_curve)

points = []
for i in range(len(source_curve)):
    if i % 2 == 1:
        points.append(source_curve[i])

scale  = 20
points = [max(1, int(i*10)) for i in points]

target_len  = fs*8

segment_len = target_len  // len(points)
half_len    = segment_len // 2
midpoints   = []
for i in range(len(points)):
    midpoints.append(segment_len*i + half_len)

chunks = {}
for m in range(len(points)):
    for p in range(points[m]):
        midpoint = midpoints[m]
        cur_point = np.random.randint(midpoint-half_len, midpoint+half_len)
        chunks[cur_point] = bloop()

output = pe.place_chunks_non_random(chunks)
wav.write('outputs/grain_cloud.wav', fs, output)














quit()


