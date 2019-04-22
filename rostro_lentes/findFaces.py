import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time

import dlib
import dlib.cuda as cuda
print(cuda.get_num_devices())
dlib.DLIB_USE_CUDA=1
dlib.USE_AVX_INSTRUCTIONS=1
image = face_recognition.load_image_file("1_18.png")

start = time()
face_locations = face_recognition.face_locations(image, model="cnn")
print("tiempo: ", time()-start)
print(format(len(face_locations)))  

i = 0

for face_location in face_locations:
    top, right, bottom, left = face_location
    #print("bla {}bla{} lba{} right{}", format(top, left, bottom, right))
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save("face-{}.png".format(i))
    i = i+1
