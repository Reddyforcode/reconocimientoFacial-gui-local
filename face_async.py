import cv2
import time
from videocaptureasync import VideoCaptureAsync
import face_recognition
from PIL import Image, ImageTk



def test(n_frames=5000, width=1280, height=720, async=False):
    

    ted_image = face_recognition.load_image_file("knowFaces/ted.png")
    #ted_image = cv2.flip(ted_image, 1)
    ted_face_encoding = face_recognition.face_encodings(ted_image)[0]

    reddy_image = face_recognition.load_image_file("knowFaces/reddy.png")
    #reddy_image = cv2.flip(reddy_image, 1)
    reddy_face_encoding = face_recognition.face_encodings(reddy_image)[0]

    obama_image = face_recognition.load_image_file("knowFaces/obama.png")
    
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    known_face_encodings = [ted_face_encoding, reddy_face_encoding, obama_face_encoding]
    known_face_names = ["Ted","Reddy", "Barack Obama"]

    face_locations =[]
    face_encodings = []
    print("loaded: ", len(known_face_encodings))

    process_this_frame = True

    
    face_names = []
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
        print("letyendo")
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #1/4 de la resolucion just for testngt
        #small_frame = cv2.flip(small_frame, 1)
        #frame = cv2.flip(frame, 1)
        #small_frame =frame
        rgb_small_frame = small_frame[:, :, ::-1]
        print("len:  ", len(known_face_encodings))
        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings =face_recognition.face_encodings(small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    print("index encontrado: ", first_match_index)
                face_names.append(name)

        process_this_frame = not process_this_frame
        a = len(face_locations)
        print(a)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            #top *= 2
            #right *= 2
            #bottom *= 2
            #left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 123), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #horizontal_img = cv2.flip( frame, 1 )
        cv2.imshow('Frame', frame)
        #cv2.imshow('horizontal', horizontal_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    print('[i] Frames per second: {:.2f}, async={}'.format(n_frames / (time.time() - t0), async))
    if async:
        cap.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test(n_frames=5000, width=1280, height=720, async=True)
    #test(n_frames=500, width=1280, height=720, async=False)
