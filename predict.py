

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.
import tensorflow as tf 
import os
import cv2
import numpy as np
from PIL import Image as img

def load_images(filenames):
  #directory = 'assn2/train'
  num= len(filenames) 
  #print(num)
  # List to store the loaded images
  images = []

  # Iterate over the files in the directory
  for i in range(num):
    #filename = filenames[i] 
    # Construct the full file path
    #print(filename)
    file_path = filenames[i] 

    # Load the image using OpenCV
    image = img.open(file_path)
    """"""
    test_image_array = np.array(image)
    red,green,blue,alpha = image.split()
    #test_image.show()
    alpha_3 = np.array(alpha)
    #alpha_3 = np.where(alpha_3==255,0,alpha_3)
    #cv2_imshow(alpha_3)
    #alpha_3 = np.where(alpha_3>0,255,alpha_3)
    #cv2_imshow(alpha_3)
    image_bgr = cv2.imread(file_path)
    # Convert BGR image to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Access individual channels (H, S, V)
    h, s, v = cv2.split(image_hsv)
    hue_channel = image_hsv[:,:,0]
    index = np.where(hue_channel==np.max(hue_channel))
    index_s = np.where(s<250)
    index_v = np.where(v<253)
    index_h = np.where(h>100)
    #index_2 = np.where(alpha_3<255)
    #alpha_3 = np.where(hue_channel==hue_channel[0][0],alplha_3=255,alpha_3)
    index_2 = np.where(test_image_array[:,:,3]<255)
    test_image_array[index_2] = [255,255,255,0]
    #cv2_imshow(test_image_array)
    test_image_array[index_s] = [255,255,255,0]
    #cv2_imshow(test_image_array)
    right_half = test_image_array[:, 368:446]
    #cv2_imshow(right_half)
    ###########shi################
    final_hsv = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
    np.unique(final_hsv[:,:,0])
    final_h, final_s, final_v = cv2.split(final_hsv)
    '''t = np.unique(final_h, return_counts=True)
    print("t:",t)
    ind = np.where(t[1]==np.max(t[1][1:]))
    #print(t[ind[0][0]])
    last_index = np.where(final_h!=t[0][ind])
    right_half[last_index] = [255]
    cv2_imshow(right_half)
    cv2_imshow(cv2.cvtColor(right_half,cv2.COLOR_BGR2GRAY))'''
    t = np.unique(final_hsv, return_counts=True)
    #print("t:",t)
    ind = np.where(t[1]==np.max(t[1][1:]))
    #print(t[ind[0][0]])
    last_index = np.where(final_hsv>200)
    right_half[last_index] = [255]
    #cv2_imshow(right_half)
    #cv2_imshow(cv2.cvtColor(right_half,cv2.COLOR_BGR2GRAY))
    # Add the image to the list
    images.append(cv2.resize(cv2.cvtColor(right_half,cv2.COLOR_BGR2GRAY),(28,28)))
  '''for i in range(num):
    images[i]=images[i][:,350:450]'''

  images= np.array(images)
  return images


def decaptcha( filenames ):
  # Invoke your model here to make predictions on the images
  loaded_model = tf.keras.models.load_model("model.h5")

  images = load_images(filenames)

  images = images /255. 

  y_preds = loaded_model.predict(images , verbose=0)>0.5

  num = len(y_preds) 
  labels = [] 
  #print(y_preds)
  for i in range(num):
    if(y_preds[i]==False):
      labels.append("ODD")
    else:
      labels.append("EVEN")


  return labels