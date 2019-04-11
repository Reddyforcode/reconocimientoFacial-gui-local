import cv2
import numpy as np 
import time
from videocaptureasync import VideoCaptureAsync
import face_recognition
from PIL import Image, ImageTk

#80 frames por segundo trhreads
def distance(accuracy):
    #pasar dos encodings procesa el nivel de accuracy de cada uno y devuelve un loading bar
    load = accuracy * 270
    color = (0, 0, 255)
    image = np.zeros((30, 300, 3), np.uint8)
    cv2.rectangle(image, (0, 0), (300, 50), (81, 88, 94), cv2.FILLED)
    cv2.rectangle(image, (10, 15), (int(load)+15, 20), color, cv2.FILLED)
    return image

def test(n_frames=5000, width=2688, height=1520, async=False):

    ted_image = face_recognition.load_image_file("knowFaces/ted.png")
    ted_face_encoding = face_recognition.face_encodings(ted_image)[0]
    
    reddy_image = face_recognition.load_image_file("knowFaces/reddy.png")
    reddy_face_encoding = face_recognition.face_encodings(reddy_image)[0]

    obama_image = face_recognition.load_image_file("knowFaces/obama.png")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    known_face_encodings = [ted_face_encoding, reddy_face_encoding, obama_face_encoding]
    known_face_names = ["Ted","Reddy", "Obama"]
    
    matches = face_recognition.compare_faces(known_face_encodings, obama_face_encoding)
    
    face_locations =[]
    face_encodings = []
    print("loaded: ", len(known_face_encodings))

    process_this_frame = True
    face_names = []
    if async:
        cap = VideoCaptureAsync()
    else:
        cap = cv2.VideoCapture()
        cap.open("rtsp://admin:DocoutBolivia@192.168.1.64:554/Streaming/Channels/102/")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if async:
        cap.start()
    t0 = time.time()
    i = 0
    nombres = {}
    face_record1 = "NADIE"
    first = True
    while i < n_frames:
        print("letyendo")
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #1/4 de la resolucion just for testngt
        rgb_small_frame = small_frame[:, :, ::-1]
        print("len:  ", len(known_face_encodings))
        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names     = []
            face_values    = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # If a match was found in known_face_encodings, just use the first one.
                values = np.linalg.norm(known_face_encodings-face_encoding, axis = 1)
                if True in matches:
                    first_match_index = matches.index(True)
                    accuracy = values[first_match_index]
                    name = known_face_names[first_match_index]
                face_names.append(name)
                face_values.append(accuracy)
        process_this_frame = not process_this_frame
        a = len(face_locations)
        print(face_names)
        print(a)
        for (top, right, bottom, left), name, acc in zip(face_locations, face_names, face_values):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            res1 = cv2.resize(frame[top:bottom, left:right], (150, 150), interpolation=cv2.INTER_CUBIC)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 123), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print("encontrado a: ", name, " con ", acc)
            
            try:
                nombres[name] = nombres[name] + 1
            except:
                nombres[name] = 1
            
            if(name!="Unknown" and nombres[name] == 10):
                if(face_record1 != name):
                    
                    path= "knowFaces/"+name.lower()+".png"
                    print(path)
                    justShow = cv2.imread(path)
                    res = cv2.resize(justShow,(150, 150), interpolation = cv2.INTER_CUBIC)
                    union = np.concatenate((res, res1), axis = 1)
                    union = np.concatenate((union, distance(acc)) , axis = 0)
                    if(first):
                        first = False
                    else:
                        final = np.concatenate((union, pastUnion), axis = 0)
                        cv2.imshow("Comparación", final)
                    face_record1 = name
                    pastUnion = union
                nombres[name] = 1
                    
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    print('[i] Frames per second: {:.2f}, async={}'.format(n_frames / (time.time() - t0), async))
    if async:
        cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test(n_frames=5000, width=1280, height=720, async=True)
    #80 fps
    #test(n_frames=5000, width=2688, height=1520, async=True)
    #130 fps
    #test(n_frames=500, width=1280, height=720, async=False)
