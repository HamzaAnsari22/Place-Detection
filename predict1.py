# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 03:52:18 2019

@author: Mohibullah
"""

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2,os
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
print("Model Loded Successfully")

path='train'
image_label=[os.path.join(path, f) for f in os.listdir(path)]
dic={}
for i in range(len(image_label)):
    dic[i]=image_label[i].split('\\')[1]
path='prediction'

n=0
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
print(str(len(image_paths))+"Files to predict")
while n!=len(image_paths):
    
    test_image = image.load_img(image_paths[n], target_size = (150,150))
    
#    print(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    #training_set.class_indices
    answer=np.argmax(result)
#    if answer == 1:
#        prediction = 'Frere Hall'
#    elif answer == 2:
#        prediction = 'Habib Bank Plaza'
#    elif answer == 3:
#        prediction = 'Shaheen Complex'
#    elif answer == 4:
#        prediction = 'Tomb of Quaid'
#    elif answer== 5:
#        prediction = 'Tower'
#    else:
#        prediction = 'Empress MArket'
#    print(prediction)
    img = cv2.imread(image_paths[n],1)
    cv2.rectangle(img, (4, 20), (150,40), (120,120,120), -1)
#    cv2.putText(img,"Hello World!!! This is ", (5,20), cv2.FONT_HERSHEY_SIMPLEX, .5, 0)
    cv2.putText(img,dic[answer], (5,35), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),2)
#    cv2.rectangle(img, (4, 5), (175,40), (255, 255, 0), -1)
#    cv2.putText(img,"Hello World!!! This is ", (5,20), cv2.FONT_HERSHEY_SIMPLEX, .5, 0)
#    cv2.putText(img,dic[answer], (5,35), cv2.FONT_HERSHEY_SIMPLEX, .5, 0,2)
    cv2.imshow('Image to Predict',img)
    k=cv2.waitKey(0)
#    print(k)
    if k==ord('q'):
        break
    elif k==44:
        n-=1
    elif k==46:
        n+=1
    if n<0:
        n=0
    elif n>len(image_paths)-1:
        n=len(image_paths)-1
#    n+=1
    


cv2.destroyAllWindows()