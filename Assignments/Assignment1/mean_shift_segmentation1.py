import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def Converge(img,UVSPACE,i,j):
	i1 = i
	j1 = j
	while(i>=0 and i<img.shape[0] and j>=0 and j<img.shape[1]):
		u = img[i,j][1]
		v = img[i,j][2]
		lefti = max(0,i-1)
		righti = min(img.shape[0],i+2)
		leftj = max(0,j-1)
		rightj = min(img.shape[1],j+2)
		arr = img[lefti:righti,leftj:rightj]
		arg = np.argmax(arr)
		if(arg%(righti-lefti)==0 and arg%(rightj-leftj)==0):
			break
		i+=1
		j+=1
	img[i1,j1] = img[i-1,j-1]


img = cv2.imread('119082.jpg',cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
# print img[0,0]
# cv2.imshow('img',img)
# print img.shape
# k = cv2.waitKey(0) & 0xFF
# if k==27:
# 	cv2.destroyAllWindows()

S = 1
m = 1

maxx=0
UVSPACE = np.zeros((256,256))
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		UVSPACE[img[i,j][1]][img[i,j][2]]+=1
		maxx = max(maxx,UVSPACE[img[i,j][1]][img[i,j][2]])
print maxx

kernel = np.array([[0.0030,0.0133,0.0219,0.0133,0.0030],
    [0.0133,0.0596,0.0983,0.0596,0.0133],
    [0.0219,0.0983,0.1621,0.0983,0.0219],
    [0.0133,0.0596,0.0983,0.0596,0.0133],
    [0.0030,0.0133,0.0219,0.0133,0.0030]])

maxx = 0
UVSPACE = cv2.filter2D(UVSPACE,-1,kernel)
for i in range(256):
	for j in range(256):
		maxx = max(maxx,UVSPACE[i,j])

print UVSPACE[200,200], maxx


# imgg = cv2.imread('119082.jpg',cv2.IMREAD_UNCHANGED)
# for i in range(imgg.shape[0]):
# 	for j in range(imgg.shape[1]):
# 		Converge(imgg,UVSPACE,i,j)

# cv2.imshow('img',imgg)
# print imgg.shape
# k = cv2.waitKey(0) & 0xFF
# if k==27:
# 	cv2.destroyAllWindows()

