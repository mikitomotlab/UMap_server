#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cassert>
#include <chrono>

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
#include <thrust/host_vector.h>

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define DILATE_ITERATIONS 2		// 線分の膨張回数
#define DATABASE_SIZE 7500      //  枚が上限    

using namespace std;
using namespace cv;

Mat cvMatQuery(const int db_cols);
Mat cvMatBinary(String filename);
cuda::GpuMat cvGpuMatBinary(String filename);
void dilateGpuMat(cuda::GpuMat d_src, cuda::GpuMat d_img);
void printDBpropaty(thrust::host_vector<Mat> dbV, vector<string> dbfilename);

typedef struct MatchingResults{
public:
	int mmid;
	double mm_pixel_ratio;
	Mat img;
}MResult;


string inputpath("H:\\dev\\workspace\\UMAP\\SEN\\XperiaXP\\2016-11-11\\img_lsd\\");
string infile;
string DBpath("H:\\dev\\workspace\\UMAP\\DB\\XperiaXP\\db0212B_0.1_ph0_dTH45\\");  //db0212B_0.1_ph0_dTH45    db0110B_0.1_ph0_dTH45
string dblist("dblist.txt");

unsigned int IMGPIXELS; // 1280*720 = 921600
unsigned int ElemSIZE;  // 1 [byte]
unsigned int DBSIZE;    //
unsigned int N;
const unsigned int NT=57600;   // 320 * 180 のピクセル数

// cudev::GlobPtrSz<uchar> pIn = cudev::globPtr(d_in.ptr<uchar>(), d_in.step, d_in.rows, d_in.cols * d_in.channels());
//__constant__ cudev::GlobPtrSz<uchar> const_d_in[NT];
//__constant__ uchar const_d_in[NT];		NT=57600 でも次のエラー 1>CUDACOMPILE : ptxas error : File uses too much global constant data (0x1b0c0 bytes, 0x10000 max)


// 画像の論理積演算
__global__ void bitwise_and(const cudev::GlobPtrSz<uchar> pIn, 
							cudev::GlobPtrSz<uchar> * ppDb, 
							cudev::GlobPtrSz<uchar> * ppLc, 
							double * d_pix, 
							const int in_pixel)
{
	//__shared__ uchar s_pixels[DATABASE_SIZE];	// 共有メモリの宣言
	//s_pixels[blockIdx.x] = 0;
	d_pix[blockIdx.x] =0;
	__syncthreads();  // 冒頭の #define __CUDACC__ にりより対処できている．
	
	const int N = pIn.rows * pIn.cols;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.z + blockIdx.z * blockDim.z;
	int tid = row * pIn.step + col;

	if (tid < N && blockIdx.x < gridDim.x ){
		if ( pIn.data[tid] ==0 || ppDb[blockIdx.x].data[tid] == 0){
			ppLc[blockIdx.x].data[tid] = 0;
		}else{
			ppLc[blockIdx.x].data[tid] = 255;
			d_pix[blockIdx.x] +=1;
		}
	}
	__syncthreads();  // 冒頭の #define __CUDACC__ にりより対処できている．

	d_pix[10] =100;

	//d_p_ratio[blockIdx.x] = 100;// (double)n_pixels[blockIdx.x]/ (double)in_pixel; 
}


MResult launchMyKernel(cuda::GpuMat d_in, cuda::GpuMat *d_dbV, cuda::GpuMat *d_lcV, const int in_pixel){
	cuda::GpuMat d_mmlc=d_dbV[0].clone();
	MResult mr;
	mr.mmid = 0;
	mr.mm_pixel_ratio = 0.0;
	//d_in.download(mr.img);

	int lc_pixels=0;
	double pixel_ratio=0;
	
	// __global__ 内では cv::cuda::GpuMat::ptr が使用できないため，型を変換
	cudev::GlobPtrSz<uchar> pIn = cudev::globPtr(d_in.ptr<uchar>(), d_in.step, d_in.rows, d_in.cols * d_in.channels());
	thrust::device_vector<cudev::GlobPtrSz<uchar>> pDb;
	thrust::device_vector<cudev::GlobPtrSz<uchar>> pLc;
	for (int i = 0; i< DATABASE_SIZE; i++){
		pDb.push_back(cudev::globPtr(d_dbV[i].ptr<uchar>(), d_dbV[i].step, d_dbV[i].rows, d_dbV[i].cols * d_dbV[i].channels()));
		pLc.push_back(cudev::globPtr(d_lcV[i].ptr<uchar>(), d_lcV[i].step, d_lcV[i].rows, d_lcV[i].cols * d_lcV[i].channels()));
	}
	// device kernel 関数のなかでは vector は使えないため，生pointer で vector を表現しなおす．
	cudev::GlobPtrSz<uchar> * ppDb = thrust::raw_pointer_cast(pDb.data());
	cudev::GlobPtrSz<uchar> * ppLc = thrust::raw_pointer_cast(pLc.data());

	double *d_p_ratio;
	cudaMalloc( (void **) d_p_ratio, sizeof(double)*DATABASE_SIZE);

	double *d_pix;
	double *h_pix;
	cudaMalloc( (void **) &d_pix, sizeof(double)*DATABASE_SIZE);
	h_pix = (double *)malloc(sizeof(double)*DATABASE_SIZE);

	// GPUの設定
    const dim3 block(1, 16, 16);
	const dim3 grid( DBSIZE, cudev::divUp(d_dbV[0].rows, block.y), cudev::divUp(d_in.cols, block.z) );

    cout << grid.x  << " " << grid.y  << " " << grid.z  << endl;
	cout << block.x << " " << block.y << " " << block.z << endl;

	// kernel 呼び出し
	bitwise_and<<<grid, block>>>(pIn, ppDb, ppLc, d_pix, in_pixel);
	cudaThreadSynchronize();

	cout << "check1" << endl;

	cudaMemcpy(h_pix, d_pix, sizeof(double)*DATABASE_SIZE, cudaMemcpyDeviceToHost);


	for (int i=0; i< DATABASE_SIZE; i++){
		pixel_ratio =  h_pix[i];// / in_pixel;
		cout << pixel_ratio << endl;
		if (mr.mm_pixel_ratio < pixel_ratio) {
			mr.mm_pixel_ratio = pixel_ratio;
			mr.mmid = i;
			d_mmlc = d_lcV[i].clone();
		}
	}


	/*
	for (int i=0; i< DATABASE_SIZE; i++){
		lc_pixels = cuda::countNonZero(d_lcV[i]);
		pixel_ratio = (double)lc_pixels / (double)in_pixel;
		if (mr.mm_pixel_ratio < pixel_ratio) {
			mr.mm_pixel_ratio = pixel_ratio;
			mr.mmid = i;
			d_mmlc = d_lcV[i].clone();
		}
	}
	*/
	
	d_mmlc.download(mr.img);
	cout << "mm_pixel_ratio[" << mr.mmid << "]: " << mr.mm_pixel_ratio << endl;
	
	return mr;
}

int main()
{
	cout << "Starting up . . ." << endl;
	auto start = chrono::system_clock::now();

//// DB image file の読み込み //////////////////////////////////////////////////
	vector<string> dbfilename;
	ifstream file(DBpath + dblist);
	string buffer;
	if (!file)
		cout << "Error: List file does not exist." << endl, exit(1);
	while (file >> buffer)
		dbfilename.push_back(buffer);

	thrust::host_vector<Mat> dbV;
	for (int i = 0;i < DATABASE_SIZE; i++) {
		dbV.push_back(imread(DBpath + dbfilename[i], IMREAD_GRAYSCALE));
	}
	printDBpropaty(dbV, dbfilename);
    
	auto int1 = chrono::system_clock::now();
	auto dur1 = int1 - start;
	auto msec1= chrono::duration_cast<chrono::milliseconds>(dur1).count();
	cout << "Reading img files: " << msec1 << " [ms]" << endl;

	//// DB image を divece メモリへ転送，LC image の領域を確保 //////////////////
	Mat Black_img;
	Black_img = dbV[0].clone();
	Black_img = cvScalar(0,0,0);

	//deviceメモリの確保
	cuda::GpuMat d_img(Black_img);
	cuda::GpuMat d_in;
	cuda::GpuMat d_dummy1, d_dummy2;
	d_in = d_img.clone();

	cuda::GpuMat d_dbV[DATABASE_SIZE];
	cuda::GpuMat d_lcV[DATABASE_SIZE];
	
	for (int i = 0;i < dbV.size(); i++) {
		d_dummy1.upload(dbV[i]);		// このような手続きでないと，device メモリ上に実体がコピーされない． 
		d_dbV[i]=d_dummy1.clone();	    //

		d_dummy2.upload(Black_img);		// このような手続きでないと，device メモリ上に実体がコピーされない． 
		d_lcV[i]=d_dummy2.clone();	    //
	}
//////////////////////////////////////////////////////////////////////////////
	auto int2 = chrono::system_clock::now();
	auto dur2 = int2 - int1;
	auto msec2= chrono::duration_cast<chrono::milliseconds>(dur2).count();
	cout << "Copy to CUDA memory: " << msec2 << " [ms]" << endl;

	while (1) {

		///// input 画像の前処理 ///////////////////////////////////////////////////////		
		Mat que = cvMatQuery(dbV[0].cols);
		int src_pixel = countNonZero(que);

		d_in.upload(que);

		cout << "call kernel " << d_in.rows << endl;
		
		auto int3 = chrono::system_clock::now();
		//// 論理積演算 プロセス /////////////////////////////////////////////////////////////
		MResult mr;
		
        mr = launchMyKernel(d_in, d_dbV, d_lcV, src_pixel);
		cout << dbfilename[mr.mmid] << endl;

		///////////////////////////////////////////////////////////////////////////
		auto int4 = chrono::system_clock::now();
		auto dur3 = int4 - int3;
		auto msec3= chrono::duration_cast<chrono::milliseconds>(dur3).count();
		cout << "Time for matching: " << msec3 << " [ms]" << endl;

		///// 結果画像の表示 ///////////////////////////////////////////////////////////////
		Mat mmlc, dst;
		Mat ddd = cvMatBinary(DBpath + dbfilename[mr.mmid]);
		vector<Mat> dilated_rgb, db_rgb, lc_rgb, integrated;

        //imshow("mr.img", mr.img);
		//waitKey(0);
        mmlc = mr.img.clone();

		// 画像の統合
		dilated_rgb.clear();
		db_rgb.clear();
		lc_rgb.clear();
		integrated.clear();

		split(que,dilated_rgb);
		split(dbV[mr.mmid], db_rgb);
		split(mmlc,lc_rgb);
		
		integrated.push_back( dilated_rgb[0] );	// 青
		integrated.push_back( db_rgb[0] );		// 緑
		integrated.push_back( lc_rgb[0] );		// 白(赤)

		merge(integrated, dst);

		imshow("result", dst);
		waitKey(0);


////////////////////////////////////////////////////////////////////////////////////
	}

}

void printDBpropaty(thrust::host_vector<Mat> dbV, vector<string> dbfilename){
	cout << "Number of imgfiles in list: " << dbfilename.size() << endl;
	cout << "Number of loaded imgfiles: " << dbV.size() << endl;

    cout << "/*--- Propaties of dbV[0] ---*/" << endl;
    cout << " rows: " << dbV[0].rows << endl;
    cout << " cols: " << dbV[0].cols << endl;
    cout << " dims: " << dbV[0].dims << endl;
    cout << " size[]: " << dbV[0].size().width << "," << dbV[0].size().height << endl;
    cout << " depth(ID): " << dbV[0].depth() <<  endl;
	cout << " channels: " << dbV[0].channels() << endl;
    cout << " elemSize: " << dbV[0].elemSize() << " [byte]" << endl;
    cout << " total: " << dbV[0].total() << " [pixels]" << endl;
    cout << " step: " << dbV[0].step << " [byte]" << endl;
    cout << "/*-------------------------*/" << endl;
    IMGPIXELS = dbV[0].total();
	ElemSIZE  = dbV[0].elemSize();
	DBSIZE    = DATABASE_SIZE;
    N = IMGPIXELS * DBSIZE;
    cout << "Number of required threads: " << N << endl; 
}

Mat cvMatQuery(const int db_cols){
	cout << "\nInput a file name: ";
	cin >> infile;
	
	Mat src = imread( inputpath + infile, IMREAD_GRAYSCALE);

	if (src.empty()) {
		cout << "Can not read source image!!" << endl;
		return src;
	}
	src = ~src;	// ネガポジ反転

	// 2値化       0 or 255 に分類   ノイズ除去のために !!!必須!!!
	Mat binarized;
	threshold(src, binarized, 120, 255, THRESH_BINARY);
	
	// 線分を膨張させる
	Mat dilated;
	dilate(binarized, dilated, Mat(), Point(-1, -1), DILATE_ITERATIONS);

	double ratio = db_cols / (double) src.cols;

	// resize
	Mat rsrc = Mat::ones(src.rows * ratio, src.cols * ratio, CV_8U);
	resize(dilated, rsrc, rsrc.size(), INTER_CUBIC);

	// 2値化       0 or 255 に分類   ノイズ除去のために !!!必須!!!
	Mat binarized2;
	threshold(rsrc, binarized2, 120, 255, THRESH_BINARY);


	return rsrc;
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

	// ネガポジ反転
	src = ~src;

	// グレースケール変換
	Mat gray_image;
	cvtColor(src, gray_image, CV_RGB2GRAY);
	
	// 2値化
	Mat binary_image;
	threshold(gray_image, binary_image, 120, 255, THRESH_BINARY_INV);

	return binary_image;
}
// 画像のバイナリ変換
cuda::GpuMat cvGpuMatBinary(String filename) {
	Mat src_image = imread(filename, cv::IMREAD_GRAYSCALE);

	if (src_image.empty()) {
		cout << "Can not read source image!!\n" << filename << endl;
		exit(1);
	}
	//GPU へ
	cout << "Upload GPU . . ." << endl;
	cuda::GpuMat d_src(src_image);

	cout << "Gray scale . . ." << endl;

	// 2値化
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
