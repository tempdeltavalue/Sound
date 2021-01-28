import numpy as np
import struct
from random import randrange



# convert to custom range
# https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio



def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, max_freq=None, eps=1e-14):
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                              shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2

    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram


def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = np.copy(image)
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)

    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = image + noise
    if clip:
        out = np.clip(out, low_clip, 1.0)

    return out

def normalize_data(data):
    min_data = np.min(data)
    max_data = np.max(data)

    if min_data == max_data:
        return np.zeros(data.shape)

    return (data - min_data) / (max_data - min_data)


# https://stackoverflow.com/questions/10848990/rgb-values-to-0-to-1-scale
# 2. https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D0%BB%D1%8C%D0%BE%D1%80%D0%B8_HTML
# https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex

def create_color_comp(amp_value, position, arr_len):
    # amp_value is float 32 here but "classicaly" we need just '2 char' per color comp (not 4 in case of float32) (2)
    res = float(amp_value * 0xffffff)
    hex_number = hex(struct.unpack('<I', struct.pack('<f', res))[0])

    comps_int_arr = get_rgba_comps_from_hex_string(hex_number)
    # comps_int_arr = [randrange(255), randrange(255), randrange(255)]
    # print(comps_int_arr)
    # if len(comps_int_arr) == 0:
    #     print('Vova is here', hex_number)
    #     print(amp_value)
    # print(hex_number)
    return comps_int_arr

def get_rgba_comps_from_hex_string(hex):
    comps_array_len = 4
    values_substring = hex[2:len(hex)] # remove 0x
    slice_amount = int(len(values_substring) / comps_array_len) #should be dividable without residue (1) (add check)
    comps = []

    # similar task in main.py for fft slicing (!)
    offset_index = 0
    while offset_index < comps_array_len * slice_amount:
        hex_slice = values_substring[offset_index:offset_index + slice_amount]

        c_number = int(hex_slice, 16) # 'doesnt work correctly' for alpha channel

        comps.append(c_number)
        offset_index += slice_amount

        # print('c_number', c_number)
        # print('slice_amount', slice_amount)
        # print('offset_index after', offset_index)

    # (!)

    # # temporary fix
    if len(comps) == 0:
        comps = [0, 0, 0, 0]

    comps[-1] /= 255 #fix alpha channel
    return comps


def remap( x, oMin, oMax, nMin, nMax ):
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result


