"""
TODO: Fix spectral leakage effect in the demo method (short click sound artifacts)
This file applies filters obtained from KEMAR database wav files spectrum
KEMAR database contains spectrum of the same sound located in different place in space
Having spectrum of such sound we can create a FIR filters (our ear pinna play a role of such kind of filter) and apply
them to our sound
The different in spectrum of left and right ear  will create an illusion that the sound in located in certain place
"""

import numpy as np
import os
from scipy.signal import lfilter
import scipy.io.wavfile as wavfile


# https://sound.media.mit.edu/resources/KEMAR.html
# https://sound.media.mit.edu/resources/KEMAR/KEMAR-FAQ.txt
# http://alumni.media.mit.edu/~kdm/hrtfdoc/section3_5.html#SECTION0005000000000000000


def readHRTF(name):
    '''Read the hrtf data from compact format files'''
    r = np.fromfile(name, np.dtype('>i2'), 256)
    r.shape = (128, 2)
    # half the rate to 22050 and scale to 0 -> 1
    r = r.astype(float)
    # should use a better filter here, this is a box lowering the sample rate from 44100 to 22050
    r = (r[0::2, :] + r[1::2, :]) / 65536
    return r


def _compute_focus_point_distance(elipse_a, elipse_b, degree, normalized=False):
    """
    Having an ellipse and a point rotated by angle get distance to focus points (focus points represent ears)
    """
    angle = degree * 2 * np.pi / 360
    a = elipse_a
    b = elipse_b
    c = np.sqrt(a ** 2 - b ** 2)
    left_focus = (-c, 0)
    right_focus = (c, 0)
    point_x = np.cos(angle) * a
    point_y = np.sqrt(b ** 2 * (1 - point_x ** 2 / a ** 2))
    if degree > 180:
        point_y = -point_y
    left_point_distance = np.sqrt((point_x - left_focus[0]) ** 2 + (point_y - left_focus[1]) ** 2)
    right_point_distance = np.sqrt((point_x - right_focus[0]) ** 2 + (point_y - right_focus[1]) ** 2)
    if normalized:
        left_point_distance /= a
        right_point_distance /= a
    return left_point_distance, right_point_distance


def locate_sound_binaural(azimuth, sound):
    """
    Locate sound usinf the shift and amplitude diff for left and right ear
    """
    N = len(sound)
    min_amp = 0.1
    distance_to_l_ear, distance_to_r_ear = _compute_focus_point_distance(5, 3, azimuth, normalized=True)

    left_amplitude = min_amp + (1 - distance_to_l_ear)
    right_amplitude = min_amp + (1 - distance_to_r_ear)

    left_phased = []
    right_phased = []

    max_shift = 300
    left_shift = int(max_shift * distance_to_l_ear)
    right_shift = int(max_shift * distance_to_r_ear)

    # Form a modulated signal from the original signal by changing phase and amplotude for different time
    for i in range(N):
        left_phased.append(left_amplitude * sound[i - left_shift])
        right_phased.append(right_amplitude * sound[i - right_shift])

    left = np.array(left_phased)
    right = np.array(right_phased)

    return left, right


def locate_sound_hrtf(elevation, azimuth, sound):
    # Compact KEMAR data stores angles only for the left ear. We need to simulate the same rotation for the right ear
    # assuming the data is symmetrical
    if azimuth <= 180:
        angle = azimuth
    else:
        angle = 360 - azimuth

    name = os.path.join('compact', f'elev{elevation}', 'H' + str(elevation) + 'e%03da.dat' % angle)
    hrtf = readHRTF(name)
    if azimuth <= 180:
        l = hrtf[:, 0]
        r = hrtf[:, 1]
    else:
        l = hrtf[:, 1]
        r = hrtf[:, 0]
    left = lfilter(l, 1.0, sound)
    right = lfilter(r, 1.0, sound)

    return left, right


def rotate_sound_horizontally(sound_mono, mode="hrtf"):
    """
    Demo for rotation sound source
    :param sound_mono:
    :param mode: hrtf/binaural
    :return:
    """
    N = len(sound_mono)
    step = 5
    chunk = int(N // (360 / step))

    elev = 0
    lefts = []
    rights = []

    i = 0
    for az in range(0, 360, step):
        i += 1
        if mode == 'hrtf':
            left, right = locate_sound_hrtf(elev, az, mono_sound[(i - 1) * chunk:i * chunk])
        else:
            left, right = locate_sound_binaural(az, mono_sound[(i - 1) * chunk:i * chunk])
        lefts.append(left)
        rights.append(right)

    left_channel = np.concatenate(lefts)
    right_channel = np.concatenate(rights)
    return left_channel, right_channel


if __name__ == '__main__':
    mode = "hrtf"
    # mode = "binaural"

    rate, mono_sound = wavfile.read('preamble.wav', 'rb')
    print("rate", rate)
    print("N", len(mono_sound))

    left_channel, right_channel = rotate_sound_horizontally(mono_sound, mode=mode)

    print("left_channel len", len(left_channel))

    result = np.array([left_channel, right_channel]).T.astype(np.int16)

    wavfile.write(mode+'_out.wav', rate, result)
