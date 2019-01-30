from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff, fs=44100, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    fs, data = wav.read('input/lil_deb.wav')

    bp = butter_bandpass_filter(data, 420.0, 460.0, fs, order=3)
    lp = butter_lowpass_filter(data, 110.0, fs, order=3)

    fn = 'lp.wav'
    wav.write(fn, fs, lp)

    wav.write('bp.wav', fs, bp)
