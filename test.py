# file: test.py
import cv2
import time
from videocaptureasync import VideoCaptureAsync



def test(n_frames=5000, width=1280, height=720, async=False):
    if async:
        cap = VideoCaptureAsync()
        print("async")

    else:
        cap = cv2.VideoCapture()
        cap.open("rtsp://admin:DocoutBolivia@192.168.1.64:554/Streaming/Channels/102/")
        print("normal")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if async:
        cap.start()
    t0 = time.time()
    i = 0
    while i < n_frames:
        _, frame = cap.read()
        cv2.imshow('Frame', frame)
        cv2.waitKey(1) & 0xFF
        i += 1
    print('[i] Frames per second: {:.2f}, async={}'.format(n_frames / (time.time() - t0), async))
    if async:
        cap.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test(n_frames=5000, width=1280, height=720, async=True)
    #test(n_frames=500, width=1280, height=720, async=False)
