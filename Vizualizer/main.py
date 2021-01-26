import librosa

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

from Vizualizer import utilities
import numpy as np
from pylab import *
from moviepy.video.io.bindings import mplfig_to_npimage
import os

#https://zulko.github.io/moviepy/getting_started/videoclips.html
import moviepy.video.io.ImageSequenceClip

#http://newt.phys.unsw.edu.au/jw/sound.spectrum.html
#https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
#https://stackoverflow.com/questions/35281427/fast-python-plotting-library-to-draw-plots-directly-on-2d-numpy-array-image-buff

def main():
    print('MAIN')
    output_name = 'video.avi'

    path = 'Sounds/ex.wav'
    samples, sample_rate = librosa.load(path)
    duration = len(samples) / sample_rate
    print('duration', duration)
    fps = 100

    print(f'Samples len {len(samples)}, sample rate {sample_rate}, duration {len(samples) / sample_rate}')


    spectrogram = utilities.spectrogram(samples,
                                       sample_rate,
                                       stride_ms=10.0,
                                       window_ms=20.0,
                                       max_freq=11000, # test selected number
                                       eps=1e-14)

    #spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    print(f'Spectogram shape {spectrogram.shape}')


    columns_count_in_one_sec = int(spectrogram.shape[1] / duration)
    print(f'columns_count_in_one_sec {columns_count_in_one_sec}')

    # plt.imshow(spectrogram[:, :])
    # plt.show()

    video_side = 2**9

    delta = int(columns_count_in_one_sec / fps)
    print('delta', delta)

    # !!!
    # TEST (uncomment lines bellow)
    # !!!

    # offset_index = 300
    # test_column = temp_func2(offset_index, spectrogram, delta, video_side)
    #
    # cv2.imshow('img', test_column)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # # # # # # #


    show_video(start_offset_index=600,
               spectrogram=spectrogram,
               video_side=video_side,
               delta=delta,
               fps=fps)


    print("FINISH")
    # out.release()


def show_video(start_offset_index, spectrogram, video_side, delta, fps):
    is_record = False

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output_video.avi',
    #                       fourcc,
    #                       fps,
    #                       (video_side, video_side))

    out_video = []
    offset_index = start_offset_index

    # prepare a small figure to embed into frame
    # frame = temp_func2(600, spectrogram, delta, video_side)
    # h, w, _ = frame.shape
    #
    # fig, ax = subplots(figsize=(4, 3), facecolor='w')
    # B = frame[:, :, 0].sum(axis=0)
    # line, = ax.plot(B, lw=7)
    # xlim([0, w])
    # ylim([30000, 40000])  # setup wide enough range here
    # box('off')
    # tight_layout()

    # graphRGB = mplfig_to_npimage(fig)
    # print("graphRGB.shape", graphRGB.shape)
    # gh, gw, _ = graphRGB.shape


    # fig = Figure()
    # canvas = FigureCanvas(fig)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2, 3, 4], [0, 6, 7, 15, 19])
    # ax.scatter([0, 1, 2, 3, 4], [1, 3, 8, 12, 27])

    plt.show()

    # while offset_index < spectrogram.shape[1]:
    #     image = temp_func2(offset_index, spectrogram, delta, video_side)
    #
    #     if is_record:
    #         temp_img = (image * 255).astype(int)
    #
    #
    #         # print(temp_img)
    #         # print(temp_img)
    #         # out.write(temp_img)
    #         out_video.append(temp_img)
    #     else:
    #         temp_img = (image * 255).astype(int)
    #
    #         # print(frame.shape)
    #         # print(frame[:, :, 0].shape)
    #         B = temp_img[:, :, 0].sum(axis=1)
    #         #
    #         # print(B)
    #         # print("w", w)
    #         # line.set_ydata(B)
    #         # image[:gh, w - gw:, :] = mplfig_to_npimage(fig)
    #         axes.plot(x, y, 'r')
    #         canvas.draw()  # draw the canvas, cache the renderer
    #
    #         image2 = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    #         cv2.imshow('frame', image)
    #         cv2.imshow('frame2', image2)
    #
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     # cv2.imshow('frame', image)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     break
    #
    #     offset_index += delta
        # print(f'offset_index {offset_index}, len {spectrogram.shape[1]}, len out_video {len(out_video)}')

    # print(np.array(out_video).shape)
    # out.release()
    # vidwrite('video.avi', np.array(out_video))
    # skvideo.io.vwrite("video.avi", out_video)

    # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(out_video, fps=fps)
    # print(clip.duration)
    # clip.write_videofile('my_video.mp4')


def temp_func2(offset_index, spectrogram, delta, video_side):
    test_columns = spectrogram[:, offset_index:offset_index + delta]
    copy_test_columns = np.zeros((test_columns.shape[0], test_columns.shape[1], 3))
    test_columns = utilities.normalize_data(test_columns)

    for ix, iy in np.ndindex(test_columns.shape):
        color_comp = utilities.create_color_comp(test_columns[ix, iy], 2, 2)
        color_comp = color_comp[0:3]
        # print('Color comps', color_comp)
        copy_test_columns[ix, iy] = color_comp

    test_columns_img = cv2.resize(copy_test_columns, (video_side, video_side))

    # (!)(!)(!) cv2 displays image correctly if we run norm AFTER resizing (It's a bug!)
    test_columns_img = utilities.normalize_data(test_columns_img)

    return test_columns_img


if __name__ == '__main__':
    main()
    # img = cv2.imread(r"C:\Users\m\Desktop\wall\wp1820012.jpg")
    # np_arr = np.array(img)
    # print(np_arr)
    # cv2.imshow("win", np_arr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


