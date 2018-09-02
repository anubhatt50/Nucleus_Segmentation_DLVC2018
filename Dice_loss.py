
# coding: utf-8

# In[ ]:


import numpy as np

def Dice_loss(img1,img2):     # img1, img2 are normalised within 0 to 1 range
    a=2*(img1*img2).sum()
    b=np.square(img1).sum()
    c=np.square(img2).sum()
    D=a/(b+c)
    return D

