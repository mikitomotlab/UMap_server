#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cassert>
#include <chrono>
#include <vector>
#include <windows.h>

//#include <winsock2.h>


#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define DILATE_ITERATIONS 1		// �����̖c����

#define PORT 20001 //�N���C�A���g�v���O�����ƃ|�[�g�ԍ������킹�Ă�������
//#define port 12357
#define DILATE_ITERATIONS 1		// �����̖c����

unsigned const int DATABASE_SIZE = 37000;// 37296;	// 320*180�̉摜�T�C�Y�ł́@40,000 ���܂�
unsigned const int IMG_ROWS = 480;
unsigned const int IMG_COLS = 270;

//using namespace std;
using namespace cv;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;

int countup_png(string path);
void listup_png(string path, string* elements, const int db_size);
Mat cvMatBinary(String filename);
cuda::GpuMat cvGpuMatBinary(String filename);
void dilateGpuMat(cuda::GpuMat d_src, cuda::GpuMat d_img);
void printDBpropaty(vector<Mat> dbV, string *  dbfs, int db_size);

string DBtableLookUp(string imgname, string tablename);

typedef struct MatchingResults {
public:
	int bmid;
	double bm_pixel_ratio;
}MResult;


string inputpath("D:/dev/workspace/UMAP/SEN/XperiaXP/samples/");
string infile;
string DBpath("D:/dev/workspace/UMAP/DB/XperiaXP/GDIDB_dbTest/");
string DBtable("C:\\dev\\workspace\\UMAP\\DB\\XperiaXP\\db0226B_0.1_ph0_dTH45.txt");

unsigned int IMGPIXELS; // 1280*720 = 921600
unsigned int ElemSIZE;  // 1 [byte]
unsigned int DBSIZE;    //
unsigned int N;         // Number of required threads = IMGPIXELS * DBSIZE


// �摜�̘_���ω��Z
//__global__ void bitwise_and(const cv::cudev::GlobPtrSz<uchar> pIn, 
//							cudev::GlobPtrSz<uchar> * ppDb, 
//							cudev::GlobPtrSz<uchar> * ppLc,
//							unsigned int * d_db_pix,
//							unsigned int * d_lc_pix, 
//							const int in_pixel )
//{
//
//	int Np = pIn.rows * pIn.cols;		// Np �̒l���Ȃ�����ʂ̑S�s�N�Z�����w���Ă��Ȃ��̂ŁC���͎g��Ȃ��D
//	int lc_pixels=0.0;
//
//	int row = threadIdx.y + blockIdx.y * blockDim.y;
//    int col = threadIdx.z + blockIdx.z * blockDim.z;
//
//    int tid = row * pIn.step + col;
//
////	if (tid < Np && blockIdx.x < gridDim.x ){
//	if ( blockIdx.x < gridDim.x ){
//		if ( pIn.data[tid] ==255 && ppDb[blockIdx.x].data[tid] == 255){
//			ppLc[blockIdx.x].data[tid] = 255;
//			atomicAdd( &d_lc_pix[blockIdx.x], 1);
//		}else{
//			ppLc[blockIdx.x].data[tid] = 0;
//		}
//	}
//
//	__syncthreads();  // �`���� #define __CUDACC__ �ɂ���Ώ��ł��Ă���D
//}



__global__ void bitwise_and(const cv::cudev::GlobPtrSz<uchar> pIn,
	cudev::GlobPtrSz<uchar> * ppDb,
	//cudev::GlobPtrSz<uchar> * ppLc,
	unsigned int * d_db_pix,
	unsigned int * d_lc_pix,
	const int in_pixel,
	unsigned int * d_lc_con)
{

	int Np = pIn.rows * pIn.cols;		// Np �̒l���Ȃ�����ʂ̑S�s�N�Z�����w���Ă��Ȃ��̂ŁC���͎g��Ȃ��D
	int lc_pixels = 0.0;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.z + blockIdx.z * blockDim.z;

	int tid = row * pIn.step + col;

	//	if (tid < Np && blockIdx.x < gridDim.x ){
	if (blockIdx.x < gridDim.x) {
		if (pIn.data[tid] == 255) {
			//ppLc[blockIdx.x].data[tid] = ppDb[blockIdx.x].data[tid];
			atomicAdd(&d_lc_con[blockIdx.x], ppDb[blockIdx.x].data[tid]);
			atomicAdd(&d_lc_pix[blockIdx.x], 1);
			//if (ppDb[blockIdx.x].data[tid] != 0) {
			//	atomicAdd(&d_lc_pix[blockIdx.x], 1);
			//}//�ގ��x�]���Q
		}
		/*else {
			ppLc[blockIdx.x].data[tid] = 0;
		}*/
	}

	__syncthreads();  // �`���� #define __CUDACC__ �ɂ���Ώ��ł��Ă���D
}


//void launchMyKernel(cuda::GpuMat d_in, cuda::GpuMat *d_dbV, cuda::GpuMat *d_lcV, const int in_pixel, int *db_nzp_V, MResult *mr){
//
//	cuda::GpuMat d_mmlc=d_dbV[0].clone();
//	mr->bmid = 0;
//	mr->bm_pixel_ratio = 0.0;
//	int lc_pixels=0;
//	double pixel_ratio=0;
//
//	// __global__ ���ł� cv::cuda::GpuMat::ptr ���g�p�ł��Ȃ����߁C�^��ϊ�
//	cudev::GlobPtrSz<uchar> pIn = cudev::globPtr(d_in.ptr<uchar>(), d_in.step, d_in.rows, d_in.cols * d_in.channels());
//	thrust::device_vector<cudev::GlobPtrSz<uchar>> pDb;
//	thrust::device_vector<cudev::GlobPtrSz<uchar>> pLc;
//	for (int i = 0; i< DATABASE_SIZE; i++){
//		pDb.push_back(cudev::globPtr(d_dbV[i].ptr<uchar>(), d_dbV[i].step, d_dbV[i].rows, d_dbV[i].cols * d_dbV[i].channels()));
//		pLc.push_back(cudev::globPtr(d_lcV[i].ptr<uchar>(), d_lcV[i].step, d_lcV[i].rows, d_lcV[i].cols * d_lcV[i].channels()));
//	}
//	// device kernel �֐��̂Ȃ��ł� vector �͎g���Ȃ����߁C��pointer �� vector ��\�����Ȃ����D
//	cudev::GlobPtrSz<uchar> * ppDb = thrust::raw_pointer_cast(pDb.data());
//	cudev::GlobPtrSz<uchar> * ppLc = thrust::raw_pointer_cast(pLc.data());
//
//	double *p_ratio;
//	cudaMalloc( (void **) &p_ratio, sizeof(double)*DATABASE_SIZE);
//
//	unsigned int *d_lc_pix;
//	unsigned int *d_db_pix;
//	cudaMalloc( (void **) &d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE);
//	cudaMalloc( (void **) &d_db_pix, sizeof(unsigned int)*DATABASE_SIZE);
//
//	unsigned int *h_lc_pix;
//	unsigned int *h_db_pix;
//	h_lc_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
//	h_db_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
//
//	for (int i=0; i<DATABASE_SIZE; i++){
//		h_lc_pix[i] = 0;
//		h_db_pix[i] = 0;
//	}
//
//	cudaMemcpy(d_lc_pix, h_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyHostToDevice);
//
//	// GPU�̐ݒ�
//    const dim3 block(1, 32, 32);
//	const dim3 grid( DBSIZE, cudev::divUp(d_in.rows, block.y), cudev::divUp(d_in.cols, block.z) );
//
//	// kernel �Ăяo��
//	bitwise_and<<<grid, block>>>(pIn, ppDb, ppLc, d_db_pix, d_lc_pix, in_pixel);
//	cudaThreadSynchronize();
//
//	cudaMemcpy(h_lc_pix, d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyDeviceToHost);
//
//	for (int i=0; i< DATABASE_SIZE; i++){
//		//lc_pixels = cuda::countNonZero(d_lcV[i]);
//		//cout << h_pix[i] << ", " << lc_pixels << endl;
//		pixel_ratio = (double)h_lc_pix[i] / (double)in_pixel;
//		if (mr->bm_pixel_ratio < pixel_ratio) {
//			mr->bm_pixel_ratio = pixel_ratio;
//			mr->bmid = i;
//			d_mmlc = d_lcV[i].clone();
//		}
//	}
//
//	cout << "bm_pixel_ratio[" << mr->bmid << "]: " << mr->bm_pixel_ratio << endl;
//}

void launchMyKernel(cuda::GpuMat d_in, cuda::GpuMat *d_dbV, /*cuda::GpuMat *d_lcV,*/ const int in_pixel, int *db_nzp_V, MResult *mr) {

	cuda::GpuMat d_mmlc = d_dbV[0].clone();
	mr->bmid = 0;
	mr->bm_pixel_ratio = 0.0;
	int lc_pixels = 0;
	double pixel_ratio = 0;

	// __global__ ���ł� cv::cuda::GpuMat::ptr ���g�p�ł��Ȃ����߁C�^��ϊ�
	cudev::GlobPtrSz<uchar> pIn = cudev::globPtr(d_in.ptr<uchar>(), d_in.step, d_in.rows, d_in.cols * d_in.channels());
	thrust::device_vector<cudev::GlobPtrSz<uchar>> pDb;
	//thrust::device_vector<cudev::GlobPtrSz<uchar>> pLc;
	for (int i = 0; i < DATABASE_SIZE; i++) {
		pDb.push_back(cudev::globPtr(d_dbV[i].ptr<uchar>(), d_dbV[i].step, d_dbV[i].rows, d_dbV[i].cols * d_dbV[i].channels()));
		//pLc.push_back(cudev::globPtr(d_lcV[i].ptr<uchar>(), d_lcV[i].step, d_lcV[i].rows, d_lcV[i].cols * d_lcV[i].channels()));
	}
	// device kernel �֐��̂Ȃ��ł� vector �͎g���Ȃ����߁C��pointer �� vector ��\�����Ȃ����D
	cudev::GlobPtrSz<uchar> * ppDb = thrust::raw_pointer_cast(pDb.data());
	//cudev::GlobPtrSz<uchar> * ppLc = thrust::raw_pointer_cast(pLc.data());

	double *p_ratio;
	cudaMalloc((void **)&p_ratio, sizeof(double)*DATABASE_SIZE);

	unsigned int *d_lc_pix;
	unsigned int *d_db_pix;
	unsigned int *d_lc_con;//�Z�xconcentration
	cudaMalloc((void **)&d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE);
	cudaMalloc((void **)&d_db_pix, sizeof(unsigned int)*DATABASE_SIZE);
	cudaMalloc((void **)&d_lc_con, sizeof(unsigned int)*DATABASE_SIZE);

	unsigned int *h_lc_pix;
	unsigned int *h_db_pix;
	unsigned int *h_lc_con;
	h_lc_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
	h_db_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
	h_lc_con = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);

	for (int i = 0; i < DATABASE_SIZE; i++) {
		h_lc_pix[i] = 0;
		h_db_pix[i] = 0;
		h_lc_con[i] = 0;
	}

	cudaMemcpy(d_lc_pix, h_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lc_con, h_lc_con, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyHostToDevice);

	// GPU�̐ݒ�
	const dim3 block(1, 32, 32);
	const dim3 grid(DBSIZE, cudev::divUp(d_in.rows, block.y), cudev::divUp(d_in.cols, block.z));

	// kernel �Ăяo��
	bitwise_and << <grid, block >> > (pIn, ppDb, /*ppLc,*/ d_db_pix, d_lc_pix, in_pixel, d_lc_con);
	cudaThreadSynchronize();

	cudaMemcpy(h_lc_pix, d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_lc_con, d_lc_con, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyDeviceToHost);

	for (int i = 0; i < DATABASE_SIZE; i++) {
		//lc_pixels = cuda::countNonZero(d_lcV[i]);
		//cout << h_pix[i] << ", " << lc_pixels << endl;
		pixel_ratio = (double)h_lc_con[i] / ((double)h_lc_pix[i]*255);




		if (mr->bm_pixel_ratio < pixel_ratio) {
			mr->bm_pixel_ratio = pixel_ratio;
			mr->bmid = i;
			//d_mmlc = d_lcV[i].clone();
		}
	}

	cout << "bm_pixel_ratio[" << mr->bmid << "]: " << mr->bm_pixel_ratio << endl;
}



int main()
{
	string str1;
	cout << "    DB directory: ";
	cin >> str1;
	if (str1 != "0") DBpath = str1;
	cout << "    Reading DB img files from ---->" << str1 << endl;

	//// DB image file �̓ǂݍ��� //////////////////////////////////////////////////
	auto STime = std::chrono::system_clock::now();
	int filenum = countup_png(DBpath);
	const int db_size = min(filenum, int(DATABASE_SIZE));

	string *dbfiles;
	dbfiles = new string[db_size];
	//delete [] dbfiles;

	listup_png(DBpath, dbfiles, db_size);

	Mat temp;
	vector<Mat> dbV;
	int db_nzp_V[DATABASE_SIZE];
	//vector<int> db_nzp_V;


	for (int i = 0; i <db_size; i++) {
		temp = imread(DBpath + "/" + dbfiles[i], 0);
		temp = ~temp;
		dbV.push_back(temp);
		db_nzp_V[i] = countNonZero(temp); /////////////////////////////
	}
	printDBpropaty(dbV, dbfiles, db_size);

	auto MTime1 = std::chrono::system_clock::now();
	auto dur1 = MTime1 - STime;
	auto msec1 = std::chrono::duration_cast<std::chrono::milliseconds>(dur1).count();
	cout << " Reading img files: " << msec1 << " [ms]" << endl;


//// DB image �� device �������֓]���CLC image �̗̈���m�� //////////////////
	Mat Black_img = Mat::zeros(dbV[0].rows, dbV[0].cols, CV_8UC1);
	//Mat dummy_img, dummy2;
	Mat dummy_img;

	//device�������̊m��
	//cuda::GpuMat d_img(Black_img), d_in(Black_img), d_db(Black_img), d_lc(Black_img), d_mmlc(Black_img);
	cuda::GpuMat d_img(Black_img), d_in(Black_img), d_db(Black_img), d_mmlc(Black_img);
	//cuda::GpuMat d_dummy1, d_dummy2;
	cuda::GpuMat d_dummy1;

	cuda::GpuMat d_dbV[DATABASE_SIZE];
	//cuda::GpuMat d_lcV[DATABASE_SIZE];

	for (int i = 0;i < dbV.size(); i++) {
		d_dummy1.upload(dbV[i]);		// ���̂悤�Ȏ葱���łȂ��ƁCdevice ��������Ɏ��̂��R�s�[����Ȃ��D 
		d_dbV[i]=d_dummy1.clone();	    //
		//d_dummy2.upload(Black_img);	// ���̂悤�Ȏ葱���łȂ��ƁCdevice ��������Ɏ��̂��R�s�[����Ȃ��D 
		//d_lcV[i]=d_dummy2.clone();	//
	}
//////////////////////////////////////////////////////////////////////////////
	auto int2 = std::chrono::system_clock::now();
	auto dur2 = int2 - MTime1;
	auto msec2 = std::chrono::duration_cast<std::chrono::milliseconds>(dur2).count();
	cout << "Copy to CUDA memory: " << msec2 << " [ms]" << endl;

	//string str2;
	//cout << "    SEN directory: ";
	//cin >> str2;
	//if (str2 != "0") inputpath = str2;

	Mat src(Black_img);

		while (1) {
			///////// TCP �ʐM �̏��� /////////////////////////////////////////////////////////////////////
			// �|�[�g�ԍ��C�\�P�b�g
			int srcSocket;  // ����
			int dstSocket;  // ����
			//sockaddr_in �\����
			struct sockaddr_in srcAddr;
			struct sockaddr_in dstAddr;
			int dstAddrSize = sizeof(dstAddr);
			int status;
			// �e��p�����[�^
			int i;
			int numrcv;
			char buffer2[1024];
			char buffer[1024];

			// Windows �̏ꍇ
			WSADATA data;
			WSAStartup(MAKEWORD(2, 0), &data);
			// sockaddr_in �\���̂̃Z�b�g
			//memset(&srcAddr, 0, sizeof(srcAddr));
			//srcAddr.sin_port = htons(PORT);
			//srcAddr.sin_family = AF_INET;
			//srcAddr.sin_addr.s_addr = htonl(INADDR_ANY);
			// �ڑ��̎�t��
			printf("---Waiting at PORT %d---\n", PORT);
			

			//cout << "check01" << endl;


				///// input �摜�̑O���� ///////////////////////////////////////////////////////		
				//cout << endl << "     Input a file name: ";
				//cin >> infile;

			memset(&srcAddr, 0, sizeof(srcAddr));
			//cout << "a" << endl;
			srcAddr.sin_port = htons(PORT);
			//cout << "b" << endl;
			srcAddr.sin_family = AF_INET;
			//cout << "c" << endl;
			srcAddr.sin_addr.s_addr = htonl(INADDR_ANY);
			//// �\�P�b�g�̐����i�X�g���[���^�j
			//srcSocket = socket(AF_INET, SOCK_STREAM, 0);
			//// �\�P�b�g�̃o�C���h
			//bind(srcSocket, (struct sockaddr *) &srcAddr, sizeof(srcAddr)); 	// std::bind �Ƃ������̂����݂��Ă���Bwinsock �� bind �Ƃ͑S���ʂ̋@�\�B
			//// �ڑ��̋���
			//listen(srcSocket, 20);

			// �\�P�b�g�̐����i�X�g���[���^�j
			//cout << "d"<< endl;
			srcSocket = socket(AF_INET, SOCK_STREAM, 0);
			//cout << "e" << endl;
			// �\�P�b�g�̃o�C���h
			bind(srcSocket, (struct sockaddr *) &srcAddr, sizeof(srcAddr)); 	// std::bind �Ƃ������̂����݂��Ă���Bwinsock �� bind �Ƃ͑S���ʂ̋@�\�B
			//cout << "f" << endl;
																				// �ڑ��̋���
			listen(srcSocket, 5000);
			//cout << "g" << endl;

			// python TCP�ʐM����̓���
			
			dstSocket = accept(srcSocket, (struct sockaddr *) &dstAddr, &dstAddrSize);
			printf("%s ����ڑ����󂯂܂���\n", inet_ntoa(dstAddr.sin_addr));

			//�p�P�b�g�̎�M
			numrcv = recv(dstSocket, buffer2, sizeof(char) * 1024, 0);
			if (numrcv == 0 || numrcv == -1) {
				status = closesocket(dstSocket); break;
			}
			/*if (numrcv == 0 || numrcv == -1) {
				status = closesocket(dstSocket);
				continue;
			}*/
			printf("Received packet:\n%s,", buffer2);

			//char c_inputpath[64], c_infile[32], c_dummy[8];
			//sscanf(buffer2, "%s %s %s", c_inputpath, c_infile, c_dummy);

			//printf("%s\n", c_inputpath);
			//printf("%s\n", c_infile);

			//inputpath = c_inputpath;
			//infile = c_infile;

			//cout << inputpath << endl;
			//cout << infile << endl;

			auto int3 = std::chrono::system_clock::now();

			src = Black_img.clone();
			//src = cvMatBinary(inputpath + "/" + infile);
			src = cvMatBinary(buffer2);

			cout << " size[]: " << src.size().width << "," << src.size().height << endl;

			// resize
			double ratio = dbV[0].cols / (double)src.cols;
			Mat rsrc = Mat::ones(src.rows * ratio, src.cols * ratio, CV_8UC1);
			resize(src, rsrc, rsrc.size(), INTER_LINEAR);
			rsrc = ~rsrc;

			cout << " size[]: " << rsrc.size().width << "," << rsrc.size().height << endl;
			imshow("in", rsrc);
			Mat aaa = dbV[1];
			//imshow("db", aaa);
			// �m�C�Y���� 0 or 255 �ɕ��� !!!�K�{!!!
			Mat bisrc;
			threshold(rsrc, bisrc, 120, 255, THRESH_BINARY);

			int src_pixel = countNonZero(bisrc);

			// ������c��������
			Mat dilated;
			dilate(bisrc, dilated, Mat(), Point(-1, -1), DILATE_ITERATIONS);
			d_in.upload(dilated);
			int d_dil_pixel = cuda::countNonZero(d_in);
			////////////////////////////////////////////////////////////////////////////////////

			cout << "call kernel" << endl;

			//// �_���ω��Z �v���Z�X /////////////////////////////////////////////////////////////
			MResult *mr;
			mr = (MResult *)malloc(sizeof(MResult));
			memset(mr, 0, sizeof(MResult));

			launchMyKernel(d_in, d_dbV, /*d_lcV,*/ src_pixel, db_nzp_V, mr);

			//Mat h_bmlcimg;
			//d_lcV[mr->bmid].download(h_bmlcimg);

			///// DBtable ���Q�Ƃ��āA X Y Z �𒊏o�@////////// 
			//double id, th, ph, ex, ey, ez, cx, cy, cz;

			//string str = DBtableLookUp(dbfiles[mr.bmid], DBtable);
			//cout << str << endl;
			//sscanf(str.data(), "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", id, th, ph, ex, ey, ez, cx, cy, cz);
			//sscanf(str.data(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &id, &th, &ph, &ex, &ey, &ez, &cx, &cy, &cz);

			//cout << "x=" << ex << " y=" << ey << " z=" << ez << " th=" << th << " ph=" << ph << endl;

			cout << "Sensing query:         " << infile << endl;
			cout << "Best matched DB image: " << dbfiles[mr->bmid] << endl;

			std::string message;
			message = dbfiles[mr->bmid];

			/*int len = sizeof(message);
			const char *m = (const char)*message;
			const char* m = message.c_str();
			int len = sizeof(m);
			send(dstSocket, m, len, 0);
			cout << "already sended" << endl;
			cout << m << endl;*/

			send(dstSocket, message.c_str(), message.length(), 0);
			cout << "alreadysended" << endl;
			cout << message << endl;

			/*const char* m = message.c_str();
			std::string str1 = m;
			int len = sizeof(str1);
			int len2 = len + 20;
			send(dstSocket, str1, len2, 0);
			cout << "alreadysended" << endl;
			cout << str1 << endl;*/

			////////////////////////////////////////////////////////////////////////////////////
			auto int4 = std::chrono::system_clock::now();
			auto dur3 = int4 - int3;
			auto msec3 = std::chrono::duration_cast<std::chrono::milliseconds>(dur3).count();
			cout << "Time for matching: " << msec3 << " [ms]" << endl;

			///// ���ʉ摜�̕\�� ///////////////////////////////////////////////////////////////
			Mat dst = Mat::ones(dbV[0].rows, dbV[0].cols, CV_8UC3);
			//Mat dst;

			vector<Mat> integrated;

			// �摜�̓���	
			//integrated.clear();
			integrated.push_back(bisrc);	     // ��
			integrated.push_back(dbV[mr->bmid]); // ��
			integrated.push_back(bisrc);	 // ��(��)
			merge(integrated, dst);

			/*
			for (int h = 0; h < dst.rows; h++) {
				for (int w = 0; w < dst.cols; w++) {
					dst.data[h * dst.elemSize() + w * dst.step + 0] =         bisrc.data[h * dst.step + w * dst.elemSize() + 0];
					//dst.data[h * dst.step + w * dst.elemSize() + 1] = dbV[mr->bmid].data[h * dst.step + w * dst.elemSize() + 0];
					//dst.data[h * dst.step + w * dst.elemSize() + 2] =     h_bmlcimg.data[h * dst.step + w * dst.elemSize() + 0];
				}
			}

			for (int h = 0; h < dst.rows; h++) {
				Vec3b *ptr = dst.ptr<Vec3b>(h);
				for (int w = 0; w < dst.cols; w++) {
					Vec3b bgr = ptr[w];

				}
			}
			*/
			imshow("db", dbV[mr->bmid]);
			imshow("result", dst);
			waitKey(2000);

			dst.release();
			for (int i = 0; i < integrated.size(); i++) integrated[i].release();
			closesocket(dstSocket);
			WSACleanup();
			////////////////////////////////////////////////////////////////////////////////////	
		}
	
}

string DBtableLookUp(string imgname, string tablename) {
	
	// �t�@�C������ .jpg ���폜
	imgname.pop_back();
	imgname.pop_back();
	imgname.pop_back();
	imgname.pop_back();

	string str;

	ifstream ifs(tablename);
	if (ifs.fail())	{
		std::cerr << "DB table open �Ɏ��s" << std::endl;
		return "-1";
	}
	while (getline(ifs, str))	{
		if (str.find(imgname) != string::npos) {
			break;
		}
	}
	return str;
}

void printDBpropaty(vector<Mat> dbV, string* dbfs, int db_size) {
	cout << "Number of imgfiles in list: " << db_size << endl;
	cout << "Number of loaded imgfiles: " << dbV.size() << endl;

	cout << "/*--- Propaties of dbV[0] ---*/" << endl;
	cout << " rows: " << dbV[0].rows << endl;
	cout << " cols: " << dbV[0].cols << endl;
	cout << " dims: " << dbV[0].dims << endl;
	cout << " size[]: " << dbV[0].size().width << "," << dbV[0].size().height << endl;
	cout << " depth(ID): " << dbV[0].depth() << endl;
	cout << " channels: " << dbV[0].channels() << endl;
	cout << " elemSize: " << dbV[0].elemSize() << " [byte]" << endl;
	cout << " total: " << dbV[0].total() << " [pixels]" << endl;
	cout << " step: " << dbV[0].step << " [byte]" << endl;
	cout << "/*-------------------------*/" << endl;
	IMGPIXELS = dbV[0].total();
	ElemSIZE = dbV[0].elemSize();
	DBSIZE = db_size;
	N = IMGPIXELS * DBSIZE;
	cout << "Number of required threads: " << N << endl;
}


Mat cvMatBinary(String filename) {
	Mat src = imread(filename);

	if (src.empty()) {
		cout << "Can not read source image!!" << endl;
		return src;
	}

	// resize
	//Mat rsrc = Mat::ones(src.rows * 0.5, src.cols * 0.5, CV_8U);
	//resize(src, rsrc, rsrc.size(), INTER_LINEAR);

	// �l�K�|�W���]
	src = ~src;

	// �O���[�X�P�[���ϊ�
	Mat gray_image;
	cvtColor(src, gray_image, CV_RGB2GRAY);

	// 2�l��
	Mat binary_image;
	threshold(gray_image, binary_image, 120, 255, THRESH_BINARY_INV);

	return binary_image;
}
// �摜�̃o�C�i���ϊ�
cuda::GpuMat cvGpuMatBinary(String filename) {
	Mat src_image = imread(filename, cv::IMREAD_GRAYSCALE);

	if (src_image.empty()) {
		cout << "Can not read source image!!\n" << filename << endl;
		exit(1);
	}
	//GPU ��
	cout << "Upload GPU . . ." << endl;
	cuda::GpuMat d_src(src_image);

	cout << "Gray scale . . ." << endl;

	// 2�l��
	cuda::GpuMat d_binary;
	cuda::threshold(d_src, d_binary, 120, 255, THRESH_BINARY_INV);
	return d_binary;
}

void dilateGpuMat(cuda::GpuMat d_src, cuda::GpuMat d_img)
{
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(2*DILATE_ITERATIONS + 1,    2*DILATE_ITERATIONS+1));
	cuda::GpuMat d_element(element);

	Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_src.type(),element);
	dilateFilter->apply(d_src,d_img);
}


int countup_png(string path) {

	int number = 0;
	HANDLE hFind;
	WIN32_FIND_DATA win32fd;

	string search_name = path + "/*.png";

	hFind = FindFirstFile(search_name.c_str(), &win32fd);

	if (hFind == INVALID_HANDLE_VALUE) throw std::runtime_error("file not found");

	do {
		if (win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
		}
		else {
			number++;
		}
	} while (FindNextFile(hFind, &win32fd));

	FindClose(hFind);

	return number;
}

void listup_png(string path, string *elements, const int db_size) {

	int num = 0;

	HANDLE hFind;
	WIN32_FIND_DATA win32fd;

	string search_name = path + "/*.png";

	hFind = FindFirstFile(search_name.c_str(), &win32fd);
	if (hFind == INVALID_HANDLE_VALUE) throw std::runtime_error("file not found");

	elements[num] = (string)win32fd.cFileName;

	while (FindNextFile(hFind, &win32fd)) {
		num++;
		if (num == db_size) break;
		elements[num] = (string)win32fd.cFileName;
		//cout << num << " " << elements[num] << endl;
	}

	FindClose(hFind);
}
