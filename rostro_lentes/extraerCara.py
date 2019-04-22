import os
#import PIL
import cv2
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time

import dlib
import dlib.cuda as cuda

path = "rostro_lente/"
i = 0
cont = 0
path_save = ""
for root,dirs,files in os.walk(path):
    i = i + 1
    for infile in [f for f in files]:
        #print("trying")
        try:
            read = path+infile
            try:
                image = face_recognition.load_image_file(read)
                face_locations = face_recognition.face_locations(image)
            except:
                print("PROBLEMA CARGANDO")
            #print("path: ", read)
            print(face_locations    )
            for face_location in face_locations:
                top, right, bottom, left = face_location
                #print("bla {}bla{} lba{} right{}", format(top, left, bottom, right))
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                try:
                    pil_image.save("justglasses/lente{}.jpg".format(cont))
                except: 
                    print("error gtuardando")

                cont = cont +1
            """for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = im[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                cont = cont + 1
                path_save = "sunglasses/sunglass_{}.jpg ".format(cont)
                pil_image.save("sunglasses/sunglass_{}.jpg ".format(cont))
                #cv2.imwrite(path_save, face_image)
                """
        except:
            print("error")