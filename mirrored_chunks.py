import numpy            as np
import pre_echos        as pe
import scipy.io.wavfile as wav




peaks, strengths = pe.get_strength_of_peaks(data, fs=44100, half_window=11025)