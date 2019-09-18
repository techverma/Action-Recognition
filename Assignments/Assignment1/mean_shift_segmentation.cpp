#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <math.h> 
#include <stack>
using namespace cv;
using namespace std;

int modes = 0, Nmodes = 0;;

double Spacial_Distance(int i,int j,int i_,int j_,int hr,int hs){
	double temp[2];
	temp[0] = i-i_;
	temp[1] = j-j_;
	double ans = 18*(temp[0]*temp[0]+temp[1]*temp[1])/(hs*hs);
	// printf("\t\t\t\t\t\tS %lf\n",ans);
	return ans;
}

double Range_Distance(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	double temp[3];
	for(int k=0;k<3;k++){
		temp[k] = (double)img.at<Vec3b>(i,j)[k]-(double)img.at<Vec3b>(i_,j_)[k];
	}
	double ans = (temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2])/(hr*hr);
	// printf("\t\t\t\t\t\tR %lf\n",ans);
	return ans;
}

double Gaussian(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	if(Spacial_Distance(i,j,i_,j_,hr,hs) > 1 || Range_Distance(img,i,j,i_,j_,hr,hs) > 1){
		return 0;
	}
	double total_distance = Spacial_Distance(i,j,i_,j_,hr,hs) + Range_Distance(img,i,j,i_,j_,hr,hs);
	// print total_distance
	return exp(-1*total_distance);
}

double Flat(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	if (Spacial_Distance(i,j,i_,j_,hr,hs) > 1 || Range_Distance(img,i,j,i_,j_,hr,hs) > 1){
		return 0;
	}
	return (Spacial_Distance(i,j,i_,j_,hr,hs) + Range_Distance(img,i,j,i_,j_,hr,hs));
}

void FindNearestMode(int*a,int*b,Mat truthValue,int M,Mat img, int hr, int hs){
	int lefti = max(0,*a-M);
	int righti = min(img.rows,*a+M+1);
	int leftj = max(0,*b-M);
	int rightj = min(img.cols,*b+M+1);
	double distance = 999999;
	double temp_distance;
	int aa = *a;
	int bb = *b;
	for(int p=lefti;p<righti;p++){
		for(int q=leftj;q<rightj;q++){
			if(truthValue.at<int>(p,q)==3){
				temp_distance = Range_Distance(img,*a,*b,p,q,hr,hs);
				if(distance > temp_distance){
					aa = p;
					bb = q;
					distance = temp_distance;
				}
			}
		}
	}
	if(*a==aa && *b==bb){
		truthValue.at<int>(*a,*b)=3;
		modes++;
	}
	else{
		truthValue.at<int>(*a,*b)=2;
		Nmodes++;
		*a = aa;
		*b = bb;
	}
}

void Converge(Mat img,Mat dest,int i,int j,int hr,int hs,double T,int iteration,Mat truthValue,int M){
	int ii = i;
	int jj = j;
	int count = 0;
	double weight;
	stack<int> backx;
	stack<int> backy;
	//modes = 0;
	while(1){
		count++;
		double num[5] = {0};
		double den = 0;
		int lefti = max(0,ii-hs);
		int righti = min(img.rows,ii+hs+1);
		// if(ii-lefti<righti-1-ii){
		// 	righti = 2*ii-lefti+1;
		// }
		// else{
		// 	lefti = 2*ii+1-righti;
		// }
		int leftj = max(0,jj-hs);
		int rightj = min(img.cols,jj+hs+1);
		// if(jj-leftj<rightj-1-jj){
		// 	rightj = 2*jj-leftj+1;
		// }
		// else{
		// 	leftj = 2*jj+1-rightj;
		// }
		for(int p=lefti;p<righti;p++){
			for(int q=leftj;q<rightj;q++){
				// printf("\t\t %d %d\n",p,q);
				weight = Gaussian(img,ii,jj,p,q,hr,hs);
				den += weight;
				num[0] = num[0] + weight*p;
				num[1] = num[1] + weight*q;
				num[2] = num[2] + weight*img.at<Vec3b>(p,q)[0];
				num[3] = num[3] + weight*img.at<Vec3b>(p,q)[1];
				num[4] = num[4] + weight*img.at<Vec3b>(p,q)[2];
			}
		}
		num[0] = num[0] / den;
		num[1] = num[1] / den;
		num[2] = num[2] / den;
		num[3] = num[3] / den;
		num[4] = num[4] / den;
		int a = (int)num[0];
		int b = (int)num[1];
		if(truthValue.at<int>(a,b)!=0){

			FindNearestMode(&a,&b,truthValue,M,img, hr, hs);

			while(!backx.empty()){
				dest.at<Vec3b>(backx.top(),backy.top())[0] = img.at<Vec3b>(a,b)[0];
				dest.at<Vec3b>(backx.top(),backy.top())[1] = img.at<Vec3b>(a,b)[1];
				dest.at<Vec3b>(backx.top(),backy.top())[2] = img.at<Vec3b>(a,b)[2];
				if(backx.top()!=a && backx.top()!=b) truthValue.at<int>(backx.top(),backy.top())=2;
				backx.pop();
				backy.pop();
			}
			break;
		}
		else{
			backx.push(ii);
			backy.push(jj);
			truthValue.at<int>(ii,jj)=1;
			ii = a;
			jj = b;
		}
	}
}

int main(){
	Mat img = imread("parrots.jpg",CV_LOAD_IMAGE_COLOR);
	Mat dest = imread("parrots.jpg",CV_LOAD_IMAGE_COLOR);
	cvtColor(img,img,CV_BGR2Luv);
	cvtColor(dest,dest,CV_BGR2Luv);
	// int hs = 10;
	// int hr = 50;
	// double T = 0.1;
	// int iterations = 20;
	int hs,hr,iterations, hello = 0, M = 20;
	double T;
	printf("INPUR PARAMS: hs hr iterations T M:\n");
	scanf("%d %d %d %lf %d",&hs,&hr,&iterations,&T,&M);
	Mat truthValue = Mat::zeros(img.rows, img.cols, CV_32F);
	for(int i=0;i<img.rows;i++){
hello = 0;
		for(int j=0;j<img.cols;j++){
			if(truthValue.at<int>(i,j)==0){
				Converge(img,dest,i,j,hr,hs,T,iterations,truthValue,M);
				//printf("%d %d \t\t %d \n",i,j,modes);
			}
else{
hello++;
//printf("%d %d \t\t %d \n",i,j,modes);
}
		}
		printf("%d %d %d %d\n",i,hello,modes,Nmodes);
	}
	// namedWindow("MyWindow", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
	// imshow("MyWindow", dest); //display the image which is stored in the 'img' in the "MyWindow" window
	// waitKey(0);  //wait infinite time for a keypress
	// destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

	cvtColor(dest,dest,CV_Luv2BGR);

	imwrite("out.jpg",dest);
	return 0;
}
// 480 0 30231 95363 <20 32 1 1>
// 
