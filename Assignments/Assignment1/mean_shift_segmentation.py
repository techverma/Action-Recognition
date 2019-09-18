import numpy as np
import cv2
import math

# def Converge(blur,Segment,i,j):
# 	while(1):
# 		if (blur[i,j])

img = cv2.imread('119082.jpg',cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
print img[0,0]
cv2.imshow('img',img)
# cv2.imshow('image',image)
print img.shape
k = cv2.waitKey(0) & 0xFF
if k==27:
	cv2.destroyAllWindows()

S = 1
m = 1

temp = np.zeros(3)
DistanceMatrix = cv2.imread('119082.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('img',DistanceMatrix)
#cv2.imshow('image',image)
k = cv2.waitKey(0) & 0xFF
if k==27:
	cv2.destroyAllWindows()
# DistanceMatrix = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		temp[0] = img[i,j][0]
		temp[1] = img[i,j][1]
		temp[2] = img[i,j][2]
		DistanceMatrix[i][j] = int(np.linalg.norm(temp)/math.sqrt(3))

blur = cv2.GaussianBlur(DistanceMatrix,(5,5),0)

cv2.imshow('img',DistanceMatrix)
#cv2.imshow('image',image)
print blur.shape
k = cv2.waitKey(0) & 0xFF
if k==27:
	cv2.destroyAllWindows()

# Segment = np.zeros((img.shape[0], img.shape[1]))
# for i in range(blur.shape[0]):
# 	for j in range(blur.shape[1]):
# 		Segment[i,j] = Converge(blur,Segment,i,j)