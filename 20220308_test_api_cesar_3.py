#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:13:43 2022

@author: cesar
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing

from numpy import asarray

import cv2


st.header("Détection de masque")
'''
Sélectionnez une photo de vous-même (ou d'une autre personne), de préférence prise de près et à l'horizontale, et ce programme vérifiera si la personne porte le masque.

'''

def main():
    file_uploaded = st.file_uploader("Choisissez un fichier", type=["jpg"])
   
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        image.save(r'loaded_image.jpg')
        fig = plt.figure()
        #plt.imshow(image)
        plt.axis("off")
        #print(image)
        #predictions = LoadAndDetectObject(image)
        LoadAndDetectObject('loaded_image.jpg')
        #LoadAndDetectObject('C:/Users/cesar/Documents/Dos/Deep Learning/projet mask 3/data/with_mask/with_mask_1323.jpg')
        #st.write(predictions)
        st.pyplot(fig)

batch_size = 40
img_height = 200
img_width = 200    

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
'data',
    validation_split=0.2,
    subset= "validation",
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

class_names = testing_ds.class_names

loaded_model = tf.keras.models.load_model('model')

def LoadAndDetectObject(image, boxSize = 100, lim = 200):
    #test_image = image.resize((200,200))
    #img = preprocessing.image.img_to_array(test_image)  
    
    
    img = plt.imread(image)
    img = cv2.resize(img,(200,200))
    numplots = (lim / boxSize)*(lim /boxSize)
    stride = int ((lim - boxSize)/ boxSize)
    
    xCurPos = 0
    yCurPos = 0
    Red = img [:,:,0]
    Blue = img[:,:,1]
    Green = img[:,:,2]
    
    y_pos = 0
    i = 0
    while y_pos < lim:
        x_pos = 0
        while x_pos < lim:
            xCurPos = x_pos
            yCurPos = y_pos
            i += 1
            if i > numplots + 1000:
                cv2.putText(img, 'No Mask', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                plt.imshow(img)
                return
            
            
            r = Red[x_pos : (x_pos + boxSize), y_pos : (y_pos + boxSize)]
            g = Blue[x_pos : (x_pos + boxSize), y_pos : (y_pos + boxSize)]
            b = Green[x_pos : (x_pos + boxSize), y_pos : (y_pos + boxSize)]
            
            ## sanity check
            xx, yy = r.shape
            if xx*yy != boxSize*boxSize:
                x_pos += stride
                continue
               
            imageSegment = cv2.merge((r,g,b))
            imageSegment  = np.array(imageSegment, dtype = 'uint8')
            imageSegment = cv2.resize(imageSegment, (lim, lim))
            result = (loaded_model.predict(np.array([imageSegment])))
            probab = max(result.flatten())
            result = (class_names[np.argmax(result)])
            #print(probab)
            
            if result == 'with_mask' and probab >= 0.8:
                probab = int (probab * 100) / 100
                img = cv2.rectangle(img, (xCurPos,yCurPos), (xCurPos + boxSize, yCurPos + boxSize), (0,255,0), 1) 
                cv2.putText(img, 'Mask', (xCurPos,yCurPos + boxSize), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                plt.imshow(img)
                return                                    
            
            x_pos += stride
        y_pos += stride
        
    cv2.putText(img, 'No Mask', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    plt.imshow(img)

  
if __name__ == "__main__":
    main() 
    
'''
Ce programme est un premier résultat du module Deep Learning de la formation de Développeur d'Intelligence Artificielle de l'Ecole IA Microsoft par Simplon.co & Greta Val de Loire.

Il est réalisé à l'aide d'un réseau neuronal convolutif.
'''        
  


        
        