import numpy as np
def transformation(array,image_size=0):

    ite = int((image_size/2)+1)
    #print(ite)
    for num in range(1, ite):
        array[(num*2)-1] = array[(num*2)-1, ::-1]

    return array

def batch_transformation(x, size=20,batch_size=128): 
     out = np.empty([batch_size,1,size,size])  
     for i in range(batch_size):
              y = np.reshape(x[i],[size,size])
              y = transformation(y,size)
              out[i,0] = y
     return out 