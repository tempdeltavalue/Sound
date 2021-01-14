import librosa

import matplotlib.pyplot as plt
import cv2

import utilities
import skvideo.io

import numpy as np

#http://newt.phys.unsw.edu.au/jw/sound.spectrum.html
#https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520




def main():
    print('MAIN')
    output_name = 'video.avi'

    path = 'Sounds/ex.wav'
    samples, sample_rate = librosa.load(path)
    duration = len(samples) / sample_rate
    fps = 24

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

    # plt.imshow(spectrogram)
    video_side = 2048

    delta = int(columns_count_in_one_sec / fps)
    print(delta)

    # !!!
    # TEST (uncomment lines bellow)
    # !!!

    offset_index = 300
    test_column = spectrogram[:, offset_index:offset_index + delta]
    copy_test_columns = np.zeros((test_column.shape[0], test_column.shape[1], 3))
    test_column = utilities.normalize_data(test_column)

    for ix, iy in np.ndindex(test_column.shape):
        copy_test_columns[ix, iy] = utilities.create_color_comp(test_column[ix, iy], 2, 2) #test_column[ix, iy]

    print(f'test_column shape {test_column.shape}')
    test_column = cv2.resize(copy_test_columns, (video_side, video_side))

    # test_column = utilities.random_noise(test_column, 'gaussian', mean=0.1, var=0.01)
    print(f'after test_column shape {test_column.shape}')

    print(test_column)

    # plt.imshow(test_column)
    # plt.show()
    # test_column = utilities.normalize_data(test_column)
    cv2.imshow('img', test_column)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # # # # # # #

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #
    # out = cv2.VideoWriter('output_video.avi',
    #                       fourcc,
    #                       fps,
    #                       (video_side, video_side))
    # offset_index = 0
    # video_len = int(spectrogram.shape[1]/delta)
    # print(video_len)
    # out_video = []
    # print(offset_index < spectrogram.shape[1])
    #
    # while offset_index < 300: #spectrogram.shape[1]:
    #     print("VOVA")
    #     test_columns = spectrogram[:, offset_index:offset_index + delta]
    #     print("test_columns shape", test_columns.shape)
    #
    #     copy_test_columns = np.zeros((test_columns.shape[0], test_columns.shape[1], 3))
    #     print("copy_test_columns shape", copy_test_columns.shape)
    #
    #     test_columns = utilities.normalize_data(test_columns)
    #
    #     for ix, iy in np.ndindex(test_columns.shape):
    #         copy_test_columns[ix, iy] = utilities.create_color_comp(test_columns[ix, iy], 2, 2) #test_columns[ix, iy]
    #
    #     test_columns_img = cv2.resize(copy_test_columns, (video_side, video_side))
    #
    #     # test_columns_img *= 255
    #     out_video.append(test_columns_img)
    #     out.write(test_columns_img)
    #
    #     # cv2.imshow('frame', test_columns_img)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     break
    #
    #     offset_index += delta
    #     print(f'offset_index {offset_index}, len {spectrogram.shape[1]}, len out_video {len(out_video)}')
    #
    # print(np.array(out_video).shape)
    # # skvideo.io.vwrite("video.mp4", np.array(out_video))
    # print("FINISH")
    # # out.release()


if __name__ == '__main__':
    main()

