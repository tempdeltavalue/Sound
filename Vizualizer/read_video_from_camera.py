import cv2
from pylab import *
from moviepy.video.io.bindings import mplfig_to_npimage

cap = cv2.VideoCapture(0)
ret, frame = cap.read() # frame is a 2D numpy array
h,w,_ = frame.shape
writer = cv2.VideoWriter( 'out.mp4', cv2.VideoWriter_fourcc('D','I','V','3'),
                fps=30, frameSize=(w,h), isColor=True )

# prepare a small figure to embed into frame
fig, ax = subplots(figsize=(4,3), facecolor='w')
B = frame[:,:,0].sum(axis=0)
line, = ax.plot(B, lw=3)
xlim([0,w])
ylim([40000, 130000]) # setup wide enough range here
box('off')
tight_layout()

graphRGB = mplfig_to_npimage(fig)
gh, gw, _ = graphRGB.shape

while True:
    ret, frame = cap.read() # frame is a 2D numpy array
    B = frame[:,:,0].sum(axis=0)
    line.set_ydata(B)
    frame[:gh,w-gw:,:] = mplfig_to_npimage(fig)

    cv2.imshow('frame', frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()