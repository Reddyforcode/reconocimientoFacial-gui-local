import cv2
import numpy as np 
from time import time 
from videocaptureasync import VideoCaptureAsync
import face_recognition
from PIL import Image, ImageTk
import threading

pastUnion = 2
first = True
path = "knowFaces/reddy.png"
db_img = cv2.imread(path)
db_img = cv2.resize(db_img, (150, 150), interpolation=cv2.INTER_CUBIC)
def distance(accuracy):
    #pasar dos encodings procesa el nivel de accuracy de cada uno y devuelve un loading bar
    load = accuracy * 270
    color = (0, 0, 255)
    image = np.zeros((30, 300, 3), np.uint8)
    cv2.rectangle(image, (0, 0), (300, 50), (81, 88, 94), cv2.FILLED)
    cv2.rectangle(image, (10, 15), (int(load)+15, 20), color, cv2.FILLED)
    return image

def dahua(name, actual_img, accuracy):
    path = "knowFaces/" + name.lower() + ".png"
    #2 images database and the one to infer
    global first
    global pastUnion
    global db_img
    #print (threading.currentThread().getName(), 'Starting')
    db_img = cv2.imread(path)
    #cv2.imshow("test", db_img)
    db_img = cv2.resize(db_img, (150, 150), interpolation=cv2.INTER_CUBIC)
    un_img = np.concatenate((db_img, actual_img), axis = 1)
    un_img = np.concatenate((un_img, distance(accuracy)), axis = 0)
    if(first):
        first = False
        cv2.imshow("Board", un_img)
    else:
        final = np.concatenate((un_img, pastUnion), axis = 0)
        cv2.imshow("Board", final)
    pastUnion = un_img
    return


#def know_face_encodings_list():
#def test(n_frames, width, height):

def test(n_frames=5000, width=2688, height=1520):

    ted_image = face_recognition.load_image_file("knowFaces/ted.png")
    ted_face_encoding = face_recognition.face_encodings(ted_image)[0]
    
    reddy_image = face_recognition.load_image_file("knowFaces/reddy.png")
    reddy_face_encoding = face_recognition.face_encodings(reddy_image)[0]

    obama_image = face_recognition.load_image_file("knowFaces/obama.png")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    known_face_encodings = [ted_face_encoding, reddy_face_encoding, obama_face_encoding]
    known_face_names = ["Ted","Reddy", "Obama"]
    
    matches = face_recognition.compare_faces(known_face_encodings, obama_face_encoding)
    
    face_locations = []
    face_encodings = []
    print("loaded: ", len(known_face_encodings))

    face_names = []
    
    cap = VideoCaptureAsync()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.start()

    nombres = {}
    face_record1 = "NADIE"
    first, process_this_frame = True, True
    accuracy = 1
    while (True):
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #1/4 de la resolucion just for testngt
        rgb_small_frame = small_frame[:, :, ::-1]   #change bgr to rgb

        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names     = []
            face_values    = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                values = np.linalg.norm(known_face_encodings-face_encoding, axis = 1)
                if True in matches:
                    first_match_index = matches.index(True)
                    accuracy = values[first_match_index]    #get the accuracy
                    name = known_face_names[first_match_index]
                face_names.append(name)
                face_values.append(accuracy)    #gui
        process_this_frame = not process_this_frame #prevent error
        a = len(face_locations)
        print("there's ",a," persons in the frame: ", face_names)
        for (top, right, bottom, left), name, acc in zip(face_locations, face_names, face_values):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            actual_img = cv2.resize(frame[top:bottom, left:right], (150, 150), interpolation=cv2.INTER_CUBIC)    #gui
            #
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 123), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #print("encontrado a: ", name, " con ", acc)
            #make sure that nombres is initializaded

            try:
                nombres[name] = nombres[name] + 1
            except:
                nombres[name] = 1
            
            if(name!="Unknown" and nombres[name] == 7):
                if(face_record1 != name):
                    start = time()
                    dahua(name, actual_img, acc)#causa 50fps con 0.02 y el mas bajo 0.001
                    print("tiempo: ", time()-start)
                    #t = threading.Thread(target = dahua, args = (name, actual_img, acc))
                    #t.start()
                    face_record1 = name
                nombres[name] = 1
                    
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test(n_frames=5000, width=1280, height=720)
    #80 fps
    #test(n_frames=5000, width=2688, height=1520, async=True)
    #130 fps
    #test(n_frames=500, width=1280, height=720, async=False)


    """
    import threading
import time
import Queue
import cv2
frames = Queue(10)

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID=ID
        self.cam=cv2.VideoCapture(ID)

    def Run(self):
        global frames
        while True:
            ret,frame=self.cam.read()
            frames.put(frame)
            time.sleep(0.1)


class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global frames
        while True:
            if(not frames.empty()):
                self.Currframe=frames.get()
            ##------------------------##
            ## your opencv logic here ##
            ## -----------------------##


grabber = ImageGrabber(0)
main = Main()

grabber.start()
main.start()
main.join()
grabber.join()
test    """
