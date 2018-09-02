
# coding: utf-8

# In[ ]:


import numpy as np
import cv2 

for m in range(1,16):    
    img=cv2.imread('/home/ranajoy/Downloads/ML DL/ML DL/challenge2/trainmask/image ('+str(m)+').png',0)
    if (m<10):
        f=open('/home/ranajoy/Downloads/ML DL/ML DL/challenge2/segmentation_training_set/image0'+str(m)+'_mask.txt','r')
    else:
        f=open('/home/ranajoy/Downloads/ML DL/ML DL/challenge2/segmentation_training_set/image'+str(m)+'_mask.txt','r')
    print(np.shape(img))
    height,width=img.shape[:2]
    x=f.readlines()
    x.pop(0)
    for i in range(len(x)):
        a=x[i]
        a=a[0:len(a)-1]
        x[i]=int(a)
    index=np.zeros((img.shape[0],img.shape[1]))
    c=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (x[c]>0):
                index[i,j]=x[c]
            c+=1
    count=0
    wt=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if img[i,j]==0:
                y=img[max(i-5,0):min(i+5,height),max(j-5,0):min(j+5,width)]
                L=np.where(y>=255)
                dist=((L[0]-5)**2+(L[1]-5)**2)**0.5
                sz=np.shape(L[0])[0]
                a=np.argsort(dist)
                if(sz>1):
                    min_x=L[0][a[0]]
                    min_y=L[1][a[0]]
                    K1=index[min_x+i-5,min_y+j-5]
                    d1=dist[a[0]]
                    k=1
                    while k<sz:
                        p=L[0][a[k]]
                        q=L[1][a[k]]
                        if(index[p+i-5,q+j-5]!=K1): 
                            d2=dist[a[k]]
                            break
                        k += 1
                    if(k<sz):    
                        wt[i,j] = 250*math.exp(-((d1+d2)**2/50)) 
                    else:
                        wt[i,j]=10
                else:
                    wt[i,j]=10
            else:
                wt[i,j]=50   

    print(np.shape(wt))
    cv2.imwrite('/home/ranajoy/Downloads/ML DL/ML DL/challenge2/wts/wt('+str(m)+').png',wt)

