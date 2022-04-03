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

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define DILATE_ITERATIONS 1		// 線分の膨張回数

unsigned const int DATABASE_SIZE = 100;// 37296;	// 320*180の画像サイズでは　40,000 枚まで


using namespace std;
using namespace cv;

Mat cvMatBinary(String filename);
cuda::GpuMat cvGpuMatBinary(String filename);
void dilateGpuMat(cuda::GpuMat d_src, cuda::GpuMat d_img);
void printDBpropaty(vector<Mat> dbV, vector<string> dbfilename);

string DBtableLookUp(string imgname, string tablename);

typedef struct MatchingResults{
public:
	int mmid;
	double mm_pixel_ratio;
	Mat img;
}MResult;


string inputpath("D:/dev/workspace/UMAP/SEN/PB2/20180913/test_noise/");
string infile;
string DBpath("D:/dev/workspace/UMAP/DB/PB2/type1_PB2_ken3_320_180_z_const/");  //db0212B_0.1_ph0_dTH45    db0110B_0.1_ph0_dTH45
string dblist("dblist.txt");
string DBtable("D:\\dev\\workspace\\UMAP\\DB\\XperiaXP\\db0226B_0.1_ph0_dTH45.txt");

unsigned int IMGPIXELS; // 1280*720 = 921600
unsigned int ElemSIZE;  // 1 [byte]
unsigned int DBSIZE;    //
unsigned int N;         // Number of required threads = IMGPIXELS * DBSIZE


// 画像の論理積演算
__global__ void bitwise_and(const cv::cudev::GlobPtrSz<uchar> pIn, 
							cudev::GlobPtrSz<uchar> * ppDb, 
							cudev::GlobPtrSz<uchar> * ppLc,
							unsigned int * d_db_pix,
							unsigned int * d_lc_pix, 
							const int in_pixel )
{

	int Np = pIn.rows * pIn.cols;		// Np の値がなぜか画面の全ピクセルを指していないので，今は使わない．
	int lc_pixels=0.0;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.z + blockIdx.z * blockDim.z;

    int tid = row * pIn.step + col;

//	if (tid < Np && blockIdx.x < gridDim.x ){
	if ( blockIdx.x < gridDim.x ){
		if ( pIn.data[tid] ==255 && ppDb[blockIdx.x].data[tid] == 255){
			ppLc[blockIdx.x].data[tid] = 255;
			atomicAdd( &d_lc_pix[blockIdx.x], 1);
		}else{
			ppLc[blockIdx.x].data[tid] = 0;
		}
	}

	__syncthreads();  // 冒頭の #define __CUDACC__ にりより対処できている．
}

MResult launchMyKernel(cuda::GpuMat d_in, cuda::GpuMat *d_dbV, cuda::GpuMat *d_lcV, const int in_pixel){

	cuda::GpuMat d_mmlc=d_dbV[0].clone();
	MResult mr;
	mr.mmid = 0;
	mr.mm_pixel_ratio = 0.0;
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

	double *p_ratio;
	cudaMalloc( (void **) &p_ratio, sizeof(double)*DATABASE_SIZE);

	unsigned int *d_lc_pix;
	unsigned int *d_db_pix;
	cudaMalloc( (void **) &d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE);
	cudaMalloc( (void **) &d_db_pix, sizeof(unsigned int)*DATABASE_SIZE);

	unsigned int *h_lc_pix;
	unsigned int *h_db_pix;
	h_lc_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
	h_db_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);

	for (int i=0; i<DATABASE_SIZE; i++){
		h_lc_pix[i] = 0;
		h_db_pix[i] = 0;
	}

	cudaMemcpy(d_lc_pix, h_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyHostToDevice);

	// GPUの設定
    const dim3 block(1, 32, 32);
	const dim3 grid( DBSIZE, cudev::divUp(d_in.rows, block.y), cudev::divUp(d_in.cols, block.z) );

    cout << grid.x  << " " << grid.y  << " " << grid.z  << endl;
	cout << block.x << " " << block.y << " " << block.z << endl;

	// kernel 呼び出し
	bitwise_and<<<grid, block>>>(pIn, ppDb, ppLc, d_db_pix, d_lc_pix, in_pixel);
	cudaThreadSynchronize();

	cout << "check1" << endl;

	cudaMemcpy(h_lc_pix, d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyDeviceToHost);


	for (int i=0; i< DATABASE_SIZE; i++){
		//lc_pixels = cuda::countNonZero(d_lcV[i]);
		//cout << h_pix[i] << ", " << lc_pixels << endl;
		pixel_ratio = (double)h_lc_pix[i] / (double)in_pixel;
		if (mr.mm_pixel_ratio < pixel_ratio) {
			mr.mm_pixel_ratio = pixel_ratio;
			mr.mmid = i;
			d_mmlc = d_lcV[i].clone();
		}
	}

	cout << "mm_pixel_ratio[" << mr.mmid << "]: " << mr.mm_pixel_ratio << endl;
	d_mmlc.download(mr.img);

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

	vector<Mat> dbV;
	for (int i = 0;i < DATABASE_SIZE; i++) {
		dbV.push_back(cvMatBinary(DBpath + dbfilename[i]));
	}
	printDBpropaty(dbV, dbfilename);

	auto int1 = chrono::system_clock::now();
	auto dur1 = int1 - start;
	auto msec1= chrono::duration_cast<chrono::milliseconds>(dur1).count();
	cout << "Reading img files: " << msec1 << " [ms]" << endl;


//// DB image を divece メモリへ転送，LC image の領域を確保 //////////////////
	Mat Black_img, dummy_img, dummy2;
	Black_img = dbV[0].clone();
	Black_img = cvScalar(0,0,0);

	//deviceメモリの確保
	cuda::GpuMat d_img(Black_img), d_in, d_db, d_lc, d_mmlc;
	cuda::GpuMat d_dummy1, d_dummy2;
	d_in   = d_img.clone();
	d_db   = d_img.clone();
	d_lc   = d_img.clone();
	d_mmlc = d_img.clone();

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
		cout << "\nInput a file name: ";
		cin >> infile;

		auto int3 = chrono::system_clock::now();

		Mat src = cvMatBinary(inputpath + infile);
		if (src.empty()) {
			cout << "Can not read source image!!\n" << inputpath  << endl;
			exit(1);
		}
		
		double ratio = dbV[0].cols / (double) src.cols;

		// resize
		Mat rsrc = Mat::ones(src.rows * ratio, src.cols * ratio, CV_8U);
		resize(src, rsrc, rsrc.size(), INTER_LINEAR);
		rsrc = ~rsrc;
		
		// ノイズ除去 0 or 255 に分類 !!!必須!!!
		Mat binarized;
		threshold(rsrc, binarized, 120, 255, THRESH_BINARY);

		int src_pixel = countNonZero(binarized);
		
		// 線分を膨張させる
		Mat dilated;
		dilate(binarized, dilated, Mat(), Point(-1, -1), DILATE_ITERATIONS);
		d_in.upload(dilated);
		int d_dil_pixel = cuda::countNonZero(d_in);
////////////////////////////////////////////////////////////////////////////////////
		
		cout << "call kernel" << endl;

//// 論理積演算 プロセス /////////////////////////////////////////////////////////////
		MResult mr;

        mr = launchMyKernel(d_in, d_dbV, d_lcV, src_pixel);


		printf("mmid: %d, mm_p_ratio: %lf\n", mr.mmid, mr.mm_pixel_ratio);
		cout << dbfilename[mr.mmid] << endl;

///// DBtable を参照して、 X Y Z を抽出　////////// 
		double id, th, ph, ex, ey, ez, cx, cy, cz;
		
		string str = DBtableLookUp(dbfilename[mr.mmid], DBtable);
		//cout << str << endl;
		//sscanf(str.data(), "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", id, th, ph, ex, ey, ez, cx, cy, cz);
		sscanf(str.data(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &id, &th, &ph, &ex, &ey, &ez, &cx, &cy, &cz);

		cout << "x=" << ex << " y=" << ey << " z=" << ez << " th=" << th << " ph=" << ph << endl;

////////////////////////////////////////////////////////////////////////////////////
		auto int4 = chrono::system_clock::now();
		auto dur3 = int4 - int3;
		auto msec3= chrono::duration_cast<chrono::milliseconds>(dur3).count();
		cout << "Time for matching: " << msec3 << " [ms]" << endl;

///// 結果画像の表示 ///////////////////////////////////////////////////////////////
		Mat mmlc, dst;
		Mat ddd = cvMatBinary(DBpath + dbfilename[mr.mmid]);
		vector<Mat> dilated_rgb, db_rgb, lc_rgb, integrated;

        mmlc = mr.img.clone();

		// 画像の統合
		dilated_rgb.clear();
		db_rgb.clear();
		lc_rgb.clear();
		integrated.clear();

		split(dilated,dilated_rgb);
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

string DBtableLookUp(string imgname, string tablename) {
	
	// ファイル名の .jpg を削除
	imgname.pop_back();
	imgname.pop_back();
	imgname.pop_back();
	imgname.pop_back();

	string str;

	ifstream ifs(tablename);
	if (ifs.fail())	{
		std::cerr << "DB table open に失敗" << std::endl;
		return "-1";
	}
	while (getline(ifs, str))	{
		if (str.find(imgname) != string::npos) {
			break;
		}
	}
	return str;
}

void printDBpropaty(vector<Mat> dbV, vector<string> dbfilename){
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

