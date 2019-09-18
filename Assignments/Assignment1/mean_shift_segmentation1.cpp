#include "opencv2/highgui/highgui.hpp" 
#include <stdio.h>
#include <math.h> 
#include <stack>
using namespace cv;
using namespace std;

double Spacial_Distance(int i,int j,int i_,int j_,int hr,int hs){
	double temp[2];
	temp[0] = i-i_;
	temp[1] = j-j_;
	double ans = (temp[0]*temp[0]+temp[1]*temp[1])/(hs*hs);
	return 0;
}

double Range_Distance(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	double temp[3];
	for(int k=0;k<3;k++){
		temp[k] = (double)img.at<Vec3b>(i,j)[k]-(double)img.at<Vec3b>(i_,j_)[k];
	}
	double ans = (temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2])/(hr*hr);
	return ans;
}

double Gaussian(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	// if(Spacial_Distance(i,j,i_,j_,hr,hs) > 1 || Range_Distance(img,i,j,i_,j_,hr,hs) > hs){
	// 	return 0;
	// }
	double total_distance = Range_Distance(img,i,j,i_,j_,hr,hs);
	return exp(-1*total_distance);
}

double Flat(Mat img,int i,int j,int i_,int j_,int hr,int hs){
	if (Spacial_Distance(i,j,i_,j_,hr,hs) > 1 || Range_Distance(img,i,j,i_,j_,hr,hs) > 1){
		return 0;
	}
	return (Spacial_Distance(i,j,i_,j_,hr,hs) + Range_Distance(img,i,j,i_,j_,hr,hs));
}

void Converge(Mat img,Mat dest,int i,int j,int hr,int hs,double T,int iteration, Mat truthValue){
	int ii = i;
	int jj = j;
	int count = 0;
	double weight = 0;
	stack<int> backx;
	stack<int> backy;
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
		// printf("%d %d %d %d %d\n",ii,jj,a,b,truthValue.at<int>(a,b));
		if(truthValue.at<int>(a,b)){
			while(!backx.empty()){
				dest.at<Vec3b>(backx.top(),backy.top())[0] = img.at<Vec3b>(a,b)[0];
				dest.at<Vec3b>(backx.top(),backy.top())[1] = img.at<Vec3b>(a,b)[1];
				dest.at<Vec3b>(backx.top(),backy.top())[2] = img.at<Vec3b>(a,b)[2];
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
	// 	if ((Range_Distance(img,ii,jj,(int)num[0],(int)num[1],hr,hs)) < T){
	// 		ii = (int)num[0];
	// 		jj = (int)num[1];
	// 		break;
	// 	}
	// 	if(ii == (int)num[0] && jj == (int)num[1]){
	// 		break;
	// 	}
	// 	ii = (int)num[0];
	// 	jj = (int)num[1];
	}
	// dest.at<Vec3b>(i,j)[0] = img.at<Vec3b>(ii,jj)[0];
	// dest.at<Vec3b>(i,j)[1] = img.at<Vec3b>(ii,jj)[1];
	// dest.at<Vec3b>(i,j)[2] = img.at<Vec3b>(ii,jj)[2];
}
	
int main(){
	char filename[] = "7-colorful-photography.jpg";
	Mat img = imread(filename,CV_LOAD_IMAGE_COLOR);
	Mat dest = imread(filename,CV_LOAD_IMAGE_COLOR);
	int hs = 20;
	int hr = 10;
	double T = 1/(hr*hr);
	int iterations = 25;
	Mat truthValue = Mat::zeros(img.rows, img.cols, CV_32F);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(!truthValue.at<int>(i,j)){
				Converge(img,dest,i,j,hr,hs,T,iterations,truthValue);
			}
		}
		printf("%d\n",i);
	}
	// namedWindow("MyWindow", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
	// imshow("MyWindow", dest); //display the image which is stored in the 'img' in the "MyWindow" window
	// waitKey(0);  //wait infinite time for a keypress
	// destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(9);

	imwrite("pramod_config_3.jpg",dest,compression_params);
	return 0;
}
