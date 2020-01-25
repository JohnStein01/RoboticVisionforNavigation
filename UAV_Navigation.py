# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:03:08 2020

@author: janberkaksu
"""

# -*- coding: utf-8 -*-
"""

This python code gets a stereo video as input and creates a disparity video, with the navigation labels defined, 
to be used in unmanned aereal and land vehicles for the purpose of obstacle avoidance

"""

import cv2
import numpy as np
import time

vidcap = cv2.VideoCapture('input_stereo_video.mp4')
success,image = vidcap.read()
count = 0
success = 0

font = cv2.FONT_HERSHEY_SIMPLEX

#Corner points of the mid-frame to be used for navigation

a=int(image.shape[0]/2)
b=int(image.shape[1]/4)

c1=int(a-200)
c2=int(b-200) 
c3=int(a+200)
c4=int(b+200)

#stereo function is set and tuned

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=145)

stereo.setPreFilterType(0)
stereo.setPreFilterCap(5)
stereo.setPreFilterSize(55)
stereo.setTextureThreshold(100)
stereo.setUniquenessRatio(0)
stereo.setSmallerBlockSize(2)

#DisparityThreshold is the threshold number that defines if a sub-frame in the mid-section is to be considered as 
#destination with an obstacle. Can be tuned to adjust sensitivity of detection.

DisparityThreshold=4000000

#image array stores the disparity images created out of every frame from stereo input video

images=[]

while count<2000:
  success,image = vidcap.read()
  im1=image[0:2*a,0:2*b,0:3]
  im2=image[0:2*a,2*b:4*b,0:3]
  gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  #disparity map is created for each frame
    
  disparity = stereo.compute(gray1,gray2) 
  
  i1=np.uint8(disparity)
        
  #Mid section of the created disparity image is cut and divided into two frames at each direction
  
  W1=i1[c1:a,c2:b]
  W2=i1[a:c3,c2:b]
  W3=i1[c1:a,b:c4]
  W4=i1[a:c3,b:c4]
  
  a1=np.sum(W1) 
  a2=np.sum(W2)
  a3=np.sum(W3)
  a4=np.sum(W4)
  
  A=np.array([a1,a2,a3,a4]) 
  C=np.array(["GO UP-LEFT!","GO UP-RIGHT!","GO DOWN-LEFT!","GO DOWN-RIGHT!"]) 
  Count=0 

  i1=cv2.applyColorMap(i1,cv2.COLORMAP_JET)
# can be switched to i1=im1 to see the navigation advises on left part of the orginal stereo image, 
#to get a better understanding of the navigation advices  
#  i1=im1
    
#6 states are created out of the 4 sub windows. The sums of each image values are calculated. If none exceeds the DisparityThreshold,
#the program tells to KEEP GOING!
#If the sum of image values on one sub window exceeds the DisparityThreshold, 
#the destination is considered as closed. Out of all 4 sub windows the one with least sum value is choosen as the way-to-go
#It can be either "GO UP-LEFT!","GO UP-RIGHT!","GO DOWN-LEFT!" or "GO DOWN-RIGHT!"
#If all sub windows are closed, the program gives the advise to STOP!
#This navigation advice is added on the image as text

  for i in range(4):
      if A[i]>DisparityThreshold:       
         Count=Count+1
   
  if Count==0:
     cv2.putText(i1,'KEEP GOING!',(c1-20,c4), font, 2,(255,255,255),2)
     print ('KEEP GOING!')  
  elif Count>0:
     d=np.argmin(A)
     cv2.putText(i1,C[d],(c1-80,c4), font, 2,(255,255,255),2)
     print (C[d])  
  elif Count==4:
      cv2.putText(i1,'STOP!',(c1-10,c4), font, 2,(255,255,255),2) 
      print ('STOP!')
      
  time.sleep(0.05)
  
  #The images are added to the list. Total number of frames can be adjusted with count variable
  
  images.append(i1)
  print ('Read a new frame: ', success)  
  count += 1
  
#After all result images with navigation advices in the list are merged into an output video. 
  
images=np.uint8(images)

cc = cv2.VideoWriter_fourcc(*'MP4V')

out = cv2.VideoWriter('output_disparity_video.mp4',cc, 20, (2*b,2*a))

 
for i in range(len(images)):
    out.write(images[i])
    
out.release()
cv2.destroyAllWindows() 
