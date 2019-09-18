import numpy as np
import cv2
import math


def Spacial_Distance(i,j,i_,j_,hr,hs):
	temp = np.zeros(2)
	temp[0] = i-i_
	temp[1] = j-j_
	ans = (temp[0]*temp[0]+temp[1]*temp[1])/(hs*hs)
	# print ans
	return ans

def Range_Distance(img,i,j,i_,j_,hr,hs):
	temp = np.zeros(3)
	temp_ = np.zeros(3)
	temp__ = np.zeros(3)
	for k in range(3):
		temp[k] = img[i,j][k]
		temp_[k] = img[i_,j_][k]
		temp__[k] = temp[k]-temp_[k]
	ans = (temp__[0]*temp__[0]+temp__[1]*temp__[1]+temp__[2]*temp__[2])/(hr*hr)
	# print ans, temp, temp_
	return ans

def Gaussian(img,i,j,i_,j_,hr,hs):
	if (Spacial_Distance(i,j,i_,j_,hr,hs) > 1 or Range_Distance(img,i,j,i_,j_,hr,hs) > 1):
		return 0
	total_distance = Spacial_Distance(i,j,i_,j_,hr,hs) + Range_Distance(img,i,j,i_,j_,hr,hs)
	# print total_distance
	return math.exp(-1*total_distance)

def Flat(img,i,j,i_,j_,hr,hs):
	if (Spacial_Distance(i,j,i_,j_,hr,hs) > 1 or Range_Distance(img,i,j,i_,j_,hr,hs) > 1):
		return 0
	return (Spacial_Distance(i,j,i_,j_,hr,hs) + Range_Distance(img,i,j,i_,j_,hr,hs))

def Converge(img,dest,i,j,hr,hs,T,iteration):
	ii = i
	jj = j
	count = 0
	while(1):
		count +=1
		# print ii,jj
		num = np.zeros(5)
		den = 0
		lefti = max(0,ii-hs)
		righti = min(img.shape[0],ii+hs+1)
		leftj = max(0,jj-hs)
		rightj = min(img.shape[1],jj+hs+1)
		for p in range(lefti,righti):
			for q in range(leftj,rightj):
				weight = Gaussian(img,ii,jj,p,q,hr,hs)
				den += weight
				num[0] += weight*p
				num[1] += weight*q
				num[2] += weight*img[p,q][0]
				num[3] += weight*img[p,q][1]
				num[4] += weight*img[p,q][2]
		if(den==0 or count>iteration):
			break;
		vector = num/den
		if ((Spacial_Distance(ii,jj,int(vector[0]),int(vector[1]),hr,hs) + Range_Distance(img,ii,jj,int(vector[0]),int(vector[1]),hr,hs)) < T):
			break;
		if(ii==int(vector[0]) and jj == int(vector[1])):
			break

		ii = int(vector[0])
		jj = int(vector[1])
	dest[i,j][0] = img[ii,jj][0]
	dest[i,j][1] = img[ii,jj][1]
	dest[i,j][2] = img[ii,jj][2]


img = cv2.imread('42049_.jpg',cv2.IMREAD_UNCHANGED)
dest = cv2.imread('42049_.jpg',cv2.IMREAD_UNCHANGED)
hs = 20	
hr = 200
T = 1
iterations = 20
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		Converge(img,dest,i,j,hr,hs,T,iterations)
	print i

cv2.imshow('img',img)
cv2.imshow('dest',dest)
k = cv2.waitKey(0) & 0xFF
if k==27:
	cv2.destroyAllWindows()

cv2.imwrite('out.jpg',dest)