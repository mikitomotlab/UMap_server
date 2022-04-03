#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cassert>
#include <chrono>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
//#include <lsd.h>
#include "lsdvcpp.h"
#include <time.h>

#include <opencv2/opencv.hpp>
#include <C:\dev\opencv-3.2.0\include\opencv\highgui.h> 

#pragma comment(lib, "opencv_highgui343.lib")
#if defined _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
//
//#pragma comment(lib, "opencv_core2413" CV_EXT)
//#pragma comment(lib, "opencv_imgproc2413" CV_EXT)
//#pragma comment(lib, "opencv_highgui2413" CV_EXT)

//#pragma comment(lib, "C:\Program Files (x86)\Windows Phone Kits\8.0\Include\ws2_32.lib")

cv::Mat img;
int n_lines;
double* lines;
//#ifdef __cplusplus
//extern "C"
//{
//#endif
//	cv::Mat image = cv::imread("image.png", 1);
//#ifdef __cplusplus
//}
//#endif

//cv::Mat change_th_lsd(int nfa, void* dummy)
cv::Mat change_th_lsd(int nfa, cv::Mat img)
{
	cv::Mat result = img.clone();
	for (int i = 0; i < n_lines; i++)
	{
		const double *line = &lines[i * 7];
		if (nfa < line[6])
		{
			const cv::Point p1(line[0], line[1]);
			const cv::Point p2(line[2], line[3]);
			cv::line(result, p1, p2, cv::Scalar(0, 0, 255));
		}
	}
	//cv::imshow("result_image", result);
	//cv::waitKey(0);
	return (result);

}

cv::Mat rotate(cv::Mat img)
{
	//cv::Mat srcImg = cv::imread("C:\Users\tmem\Desktop\Shirahama2018\Shirahama2018\1_30\sen\image.png");
	cv::Mat srcImg = img;
	/*if (srcImg.empty())
		return 1;*/

	cv::Point2f center = cv::Point2f(
		static_cast<float>(srcImg.cols / 2),
		static_cast<float>(srcImg.rows / 2));

	double degree = 45.0;  // ��]�p�x
	double scale = 1.0;    // �g�嗦

	// �A�t�B���ϊ��s��
	cv::Mat affine;
	cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);

	cv::warpAffine(srcImg, srcImg, affine, srcImg.size(), cv::INTER_CUBIC);

	cv::namedWindow("image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("image", srcImg);
	cv::waitKey();

	return (srcImg);
}



int main()
{
	cv::Mat img = cv::imread("181220160136raspi7.jpg",1); 
	//cv::Mat image(480, 270, CV_32F, img);
	cv::imshow("sensingimage", img);
	cv::waitKey(5000);
	//cv::imwrite("D:\\dev\\workspace\\UMap\\SEN\\PiCA\\1.img\\'+ rcvmsg+'.jpg", image);

	//LSD�p�摜�ɕϊ�����
	double *dat = new double[img.rows * img.cols];
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			dat[y * img.cols + x] = img.at<unsigned char>(y, x);

	//LSD����
	lines = lsd(&n_lines, dat, img.cols, img.rows);
	cv::imshow("result_image", img);
	cv::waitKey(5000);

	//�������l�̍ő�l�ƍŏ��l�������Ă���
	int max_NFA = 0;
	/*for (int i = 0; i < n_lines; i++)
		max_NFA = std::max(max_NFA, static_cast<int>(lines[i * 7 + 6]));
	cv::imshow("result_image", img);
	cv::waitKey(5000);
*/
	//���ʕ`��p�摜
	//cv::cvtColor(img, img, CV_GRAY2RGB);

	//���ʕ\���p�E�B���h�E
	//cv::namedWindow("result_image");
	//cv::createTrackbar("NFA", "result_image", NULL, max_NFA, change_th_lsd);
	//cv::setTrackbarPos("NFA", "result_image", max_NFA);
	img = change_th_lsd(max_NFA, img);
	//���ʕ\��
	cv::imshow("result_image", img);
	cv::waitKey(5000);

	img = rotate(img);
	//���ʕ\��
	cv::imshow("result_image", img);
	cv::waitKey(5000);


		

	}



