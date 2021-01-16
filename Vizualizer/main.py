import librosa

import matplotlib.pyplot as plt
import cv2

import utilities
import skvideo.io
import ffmpeg
import numpy as np

#http://newt.phys.unsw.edu.au/jw/sound.spectrum.html
#https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

def main():
    print('MAIN')
    output_name = 'video.avi'

    path = 'Sounds/ex.wav'
    samples, sample_rate = librosa.load(path)
    duration = len(samples) / sample_rate
    fps = 48 * 2

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
    video_side = 2**9

    delta = int(columns_count_in_one_sec / fps)
    print(delta)

    # !!!
    # TEST (uncomment lines bellow)
    # !!!
    #
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
    is_record = True

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output_video.avi',
    #                       fourcc,
    #                       fps,
    #                       (video_side, video_side))

    out_video = []
    offset_index = start_offset_index

    while offset_index < spectrogram.shape[1]:
        image = temp_func2(offset_index, spectrogram, delta, video_side)

        if is_record:
            out_video.append(image)
            # out.write(image)
        else:
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        offset_index += delta
        print(f'offset_index {offset_index}, len {spectrogram.shape[1]}, len out_video {len(out_video)}')

    # print(np.array(out_video).shape)
    # out.release()
    vidwrite('video.avi', np.array(out_video))
    # skvideo.io.vwrite("video.avi", out_video)


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

def vidwrite(fn, images, framerate=60, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


if __name__ == '__main__':
    main()
    # img = cv2.imread(r"C:\Users\m\Desktop\wall\wp1820012.jpg")
    # np_arr = np.array(img)
    # print(np_arr)
    # cv2.imshow("win", np_arr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


