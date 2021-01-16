import numpy as np
import cv2


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc, 48 * 2, (2**9,2**9))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

    while (cap.isOpened()):
        # ret, frame = cap.read()
        frame = cv2.imread(r"C:\Users\m\Desktop\wall\wp1820012.jpg")
        if True:
            frame = cv2.flip(frame, 0)
            frame = cv2.resize(frame, (2**9, 2**9))

            # write the flipped frame
            print(np.array(frame))
            out.write(np.array(frame))

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
