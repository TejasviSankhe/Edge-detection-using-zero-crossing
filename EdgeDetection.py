import cv2
import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib as mp

def filterApply(x,y):
    return sg.convolve2d(x,y)
   
temp = cv2.imread('H:\Fall 16\CVIP-CS573\HW\HW4\UBCampus.jpg')
ub = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
ubw = np.size(ub,0)
ubh = np.size(ub,1)

filterDoG=[[0.0,0.0,-1.0,-1.0,-1.0,0.0,0.0],
        [0.0,-2.0,-3.0,-3.0,-3.0,-2.0,0.0],
        [-1.0,-3.0,5.0,5.0,5.0,-3.0,-1.0],
        [-1.0,-3.0,5.0,16.0,5.0,-3.0,-1.0],
        [-1.0,-3.0,5.0,5.0,5.0,-3.0,-1.0],
        [0.0,-2.0,-3.0,-3.0,-3.0,-2.0,0.0],
        [0.0,0.0,-1.0,-1.0,-1.0,0.0,0.0]]

#Dog=np.absolute(Dog)
#Dog=(Dog-np.min(Dog))/float(np.max(Dog)-np.min(Dog))

Dog=filterApply(ub,filterDoG)
cv2.imshow('DoG',Dog)
#sp.misc.toimage(Dog).save('H:\Fall 16\CVIP-CS573\HW\HW4\DOGApplied.jpg')

emptyImage = np.zeros(shape=(ubw, ubh), dtype=int)
zeroCross = np.array(emptyImage)


for i in range (1,ubw):
   for j in range (1,ubh):
       count=0
       if (((Dog[i-1,j]>0 and Dog[i,j]<0) or (Dog[i-1,j]<0 and Dog[i,j]>0)) or ((Dog[i,j-1]>0 and Dog[i,j]<0) or (Dog[i,j-1]<0 and Dog[i,j]>0))):
          count+=1
       elif (((Dog[i+1,j]>0 and Dog[i,j]<0) or (Dog[i+1,j]<0 and Dog[i,j]>0)) or ((Dog[i,j+1]>0 and Dog[i,j]<0) or (Dog[i,j+1]<0 and Dog[i,j]>0))):
          count+=1
       
       if(count>0):
           zeroCross[i,j]=0
       else:
            zeroCross[i,j]=1

#sp.misc.toimage(zeroCross).save("H:\Fall 16\CVIP-CS573\HW\HW4\DogZeroCross.jpg")
sp.misc.toimage(zeroCross).show()

ub = ub.astype('int32')
Gx = sp.ndimage.sobel(ub, 1)  
Gy = sp.ndimage.sobel(ub, 0)  
magnitude = np.hypot(Gx,Gy) 
sp.misc.imsave('H:\Fall 16\CVIP-CS573\HW\HW4\sobel.jpg',magnitude)
mag=cv2.imread("H:\Fall 16\CVIP-CS573\HW\HW4\sobel.jpg")
ret,thresh = cv2.threshold(mag,27,255,cv2.THRESH_BINARY)
#sp.misc.imsave('H:\Fall 16\CVIP-CS573\HW\HW4\sobelThresh.jpg',thresh)

#mp.pyplot.imshow(thresh)
#mp.pyplot.show()


DOGEdge= [[0.0 for x in range(ubh)] for y in range(ubw)]

for i in range(1, ubw):
        for j in range(1, ubh):
            DOGEdge[i][j] = abs(thresh[i][j][0] - zeroCross[i][j])


sp.misc.toimage(DOGEdge).show()
#sp.misc.toimage(DOGEdge).save('H:\Fall 16\CVIP-CS573\HW\HW4\DOGEdges.jpg')

filterLoG=[[0.0,0.0,1.0,0.0,0.0],
        [0.0,1.0,2.0,1.0,0.0],
        [1.0,2.0,-16.0,2.0,1.0],
        [0.0,1.0,2.0,1.0,0.0],
        [0.0,0.0,1.0,0.0,0.0]]
        
Log=filterApply(ub,filterLoG)
cv2.imshow('LoG',Log)
#cv2.imwrite('H:\Fall 16\CVIP-CS573\HW\HW4\LoGApplied.jpg',Log)

emptyImage = np.zeros(shape=(ubw, ubh), dtype=int)
zeroCrossLog = np.array(emptyImage)


for i in range (1,ubw):
   for j in range (1,ubh):
       count=0
       if (((Log[i-1,j]>0 and Log[i,j]<0) or (Log[i-1,j]<0 and Log[i,j]>0)) or ((Log[i+1,j]>0 and Log[i,j]<0) or (Log[i+1,j]<0 and Log[i,j]>0))) :
          count+=1
       elif (((Log[i,j-1]>0 and Log[i,j]<0) or (Log[i,j-1]<0 and Log[i,j]>0)) or ((Log[i,j+1]>0 and Log[i,j]<0) or (Log[i,j+1]<0 and Log[i,j]>0))):
          count+=1
       
       if(count>0):
           zeroCrossLog[i,j]=0
       else:
            zeroCrossLog[i,j]=1

#sp.misc.toimage(zeroCrossLog).save("H:\Fall 16\CVIP-CS573\HW\HW4\LogZeroCross.jpg")
sp.misc.toimage(zeroCrossLog).show()
#mp.pyplot.show()


LOGEdge= [[0.0 for x in range(ubh)] for y in range(ubw)]

for i in range(1, ubw):
        for j in range(1, ubh):
            LOGEdge[i][j] = abs(thresh[i][j][0] - zeroCrossLog[i][j])


sp.misc.toimage(LOGEdge).show()
#sp.misc.toimage(LOGEdge).save('H:\Fall 16\CVIP-CS573\HW\HW4\LOGEdges.jpg')
