import numpy            as np
import fx_functions        as pe
import scipy.io.wavfile as wav
import librosa          as lib
from multiprocessing import Process




if __name__ == '__main__':
    input_fn = 'input/little_deb.wav'
    fs, data = wav.read(input_fn)
    #get onsets
    onsets   = lib.onset.onset_detect(y=data, sr=fs, hop_length=512, units='samples', backtrack=True)

    #estimate strengths of each onset
    peaks, strengths = pe.get_strength_of_peaks(data, onsets, fs)

    #keep only the strongest peaks
    pairs            = pe.winnow_peaks(peaks, strengths, 0.5)
    strengths, peaks = zip(*pairs)  #split pairs into separate lists

    #get chunks associated with peaks and add envelope to each
    peaks, chunks    = pe.get_chunks(data, peaks)
    chunks           = pe.apply_all_envelopes(chunks)

    procs = []
    for p in range(5):
        procs.append(Process(target=pe.mirrored_chunks, args=(chunks, peaks, strengths, len(data), p, fs, False)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()