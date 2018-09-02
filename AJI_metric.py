
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling

#IoU
def iou(gt,pred):
    inter=gt*pred
    union=gt+pred-inter
    inter=inter.sum()
    union=union.sum()
    return inter,union

#AJI metric
def AJI_metric(img1,img2):
    #image 1
    img1=img1[:,:,0]
    img1=cv2.resize(img1, (256,256)) 
    labels1 = skimage.morphology.label(img1,neighbors=8)
    max1=np.max(labels1)
    #image 2
    img2=img2[:,:,0]
    img2=cv2.resize(img2, (256,256)) 
    labels2 = skimage.morphology.label(img2,neighbors=8)
    max2=np.max(labels2)
    #metric
    out=[]
    Intersection=0
    Union=0
    for i in range(1,max1+1):
        List=[]
        gt=np.zeros((256,256))
        L=np.where(labels1==i)
        for j in range(len(L[0])):
            gt[L[0][j],L[1][j]]=1.0
            x=labels2[L[0][j],L[1][j]]
            if(x>0 and (x not in List) and (x not in out) ):
                List.append(x)
        maxm=0
        maxinter=0
        maxun=0
        maxidx=0  
        if(len(List)>0):
            for p in List:
                pred=np.zeros((256,256))
                L1=np.where(labels2==p)
                for q in range(len(L1[0])):
                    pred[L1[0][q],L1[1][q]]=1.0
                inter,union=iou(gt,pred)
                val=inter/union
                if(val>maxm):
                    maxm=val
                    maxinter=inter
                    maxun=union
                    maxidx=p 
            if(maxidx>0): 
                out.append(maxidx) 
                Intersection = Intersection + maxinter
                Union = Union + maxun

    for i in range(max2+1):
        if(i not in out):
            L1=np.where(labels2==p)
            for q in range(len(L1[0])):
                pred[L1[0][q],L1[1][q]]=1.0
            Union = Union+pred.sum()

    if(Union>0):
        return (Intersection/Union)
    else:
        print("Union=0")

