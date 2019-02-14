from __future__ import division
import random 
import numpy as np
from skimage.io import imread
from scipy.spatial import distance
from matplotlib import pyplot as plt 

import string as str




def imgintovecs(path):
    X = np.zeros([26*7,144])
    chars=[]
    for char in str.ascii_lowercase:
        chars.append(char)
    i=0
    for letter in range(0,26):
        letter=chars[letter]
        for index in range(1,8):
                x="%sA1%s%d.jpg"%(path,letter,index)
                img = imread(x)
                img=img.flatten()
                X[i,:] = img 
                i=i+1
    return X


#
#def labeling(a):
#    chars=[]
#    labels=[]
#    index=[]
#    for char in str.ascii_lowercase:
#            chars.append(char)
#    for i in range(0,26):
#            for k in range(1,8):
#                labels.append(ord(chars[i]))
#                index.append(k)
#    b=np.array(labels)
#    c =np.column_stack((a,b))
#    labeled =np.column_stack((c,index))
#    return labeled

def allEucl(data):
    dis=np.zeros([182,182])
    for i in range(0,182):
        for j in range(0,182):
            dis[i,j]=distance.euclidean(data[i], data[j])
    return dis

#def findcenters(dismatrix, numofcenters, data):
#    dump=data
#    centers=[]
#    indecies = range(0, 182)
#    index=random.sample(indecies, 1)[0]
#    centers.append(index)
#    while len(centers) < numofcenters:
#        for i in range(len(dismatrix[index])):
#            if max(dismatrix[index]) == dismatrix[index][i]:
#                if i not in centers:
#                    centers.append(i)
#                    index=i
#                else:
#                    dismatrix[index][i]=0
#                    index=index
#    if len(centers)>numofcenters:
#        centers.remove(centers[-1])
##    x=labeling(data)
#    centersdata=np.zeros([ len(centers), len(data[0])])
#    for i in range(len(centers)):
#        centersdata[i]=dump[centers[i]]
#
##        print '%s%d' %(chr(int(x[centers[i]][-2])) , int(x[centers[i]][-1]))
#    return centersdata 



def pointcluterdistances(centers, points):
    dis=np.zeros([len(points), len(centers)])
    for i in range(len(points)):
        for j in range(len(centers)):
            dis[i,j]=distance.euclidean( points[i], centers[j])
    return dis

#print [item for item, count in collections.Counter(c).items() if count > 1]

def safe_div(x,y):
    if y == 0 or x ==0:
        return 0
    return x/float( y)
  
def MembershipMatrix( q, centers, points  ): 
   num=len(centers)
   dis=pointcluterdistances(centers, points) 
   Membership=np.zeros([182,26])
   p=float(2.0/(q-1.0))
   for i in range(len(dis)):
       for j in range(num):
           den=sum([float(np.power(safe_div(dis[i][j],dis[i][k]),p)) for k in range(num)])
           Membership[i][j]= safe_div(1, den)
   return Membership

def sumColumn(m):
    return [sum(col) for col in zip(*m)]

def isone(x):
    for i in range(7,len(x)):
        
        if float(sum(x[i])) != 1:
            print i
            return False
    return True

                   
def newcentriods(oldcentroid, memebership, data, q):
    newcentriods=np.zeros([26,144]) 
    memebership=memebership.transpose()                
    for i in range(26):
        num=sum([np.power((memebership[i][k]),q) * data[k] for k in range(182)])
        den=sum([float(np.power((memebership[i][k]),q)) for k in range(182)])
          
        newcentriods[i]=(num/den) 
    return newcentriods


def shouldstop(x, y):
    l=[]
    for i in range(len(x)):
        l.append(distance.euclidean(y[i], x[i]))
    avg=sum(l)/float(len(l))
    if avg < 0.00001:
        return True
    else:
        return False


def Finalize(Finalle):
    for i in range(182):
        for j in range(26):
            if max(Finalle[i])== Finalle[i][j]:
                Finalle[i][j]=1
            else:
                Finalle[i][j]=0
    return Finalle


def initialCenters(data):
    dis=[]
    centers=[]
    indecies = range(0, 182)
    index=random.sample(indecies, 1)[0]
    centers.append(index)

    for j in range(0,182):
        dis.append(distance.euclidean(data[index], data[j]))
    second=np.argmax(dis)
    centers.append(second)
    centers_data=np.zeros([2,144])
    centers_data[0]=data[centers[0]]
    centers_data[1]=data[centers[1]]
    
    while len(centers) < 26:
        centers=farestpoint(centers, centers_data,  data)
        centers_data=np.vstack([centers_data, data[centers[-1]]])
    
    return centers_data
 
                
def vectorow(x):
    y=np.zeros([1,144])
    for i in range(144):
        y[0][i]=x[i]
    return y

   
   
def farestpoint(centers, cent, data):
    distances=np.zeros([182,])
   
    for i in range(182):
        distances[i]=np.linalg.norm( cent - data[i])
        
    while True:
        for i in range(len(distances)):
            if max(distances)== distances[i]:
                latest=i

        if latest not in centers:
            centers.append(latest)
            break
        else:
            distances[latest]=0
             

    return centers




x=imgintovecs('Assignment 3 Dataset/')

MainDisMatrix=allEucl(x)

centers=initialCenters(x)

M=MembershipMatrix(1.25, centers, x  ) 



while True:
     
    N=newcentriods(centers, M, x, 1.25)
    if shouldstop(centers,N ):
        print 'done!'
        break
    else:
        centers=N
        M=MembershipMatrix(1.25, N, x  )  
      
F=Finalize(M)

ff=sumColumn(F)

plt.plot(ff)

plt.savefig('Accuracy.png')        

    
                   

       


            
       






















