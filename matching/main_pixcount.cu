//csv吐き出し　ピクセルカウント

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cassert>
#include <chrono>
#include <vector>
#include <windows.h>

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

unsigned const int DATABASE_SIZE = 37000;// 37296;	// 320*180の画像サイズでは　40,000 枚まで
unsigned const int IMG_ROWS = 180;
unsigned const int IMG_COLS = 320;

using namespace std;
using namespace cv;

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


// 画像の論理積演算
//__global__ void bitwise_and(const cv::cudev::GlobPtrSz<uchar> pIn, 
//							cudev::GlobPtrSz<uchar> * ppDb, 
//							cudev::GlobPtrSz<uchar> * ppLc,
//							unsigned int * d_db_pix,
//							unsigned int * d_lc_pix, 
//							const int in_pixel )
//{
//
//	int Np = pIn.rows * pIn.cols;		// Np の値がなぜか画面の全ピクセルを指していないので，今は使わない．
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
//	__syncthreads();  // 冒頭の #define __CUDACC__ にりより対処できている．
//}



// 画像の論理積演算
__global__ void bitwise_and(const cv::cudev::GlobPtrSz<uchar> pIn,
	cudev::GlobPtrSz<uchar> * ppDb,
	cudev::GlobPtrSz<uchar> * ppLc,
	unsigned int * d_db_pix,
	unsigned int * d_lc_pix,
	const int in_pixel)
{

	int Np = pIn.rows * pIn.cols;		// Np の値がなぜか画面の全ピクセルを指していないので，今は使わない．
	int lc_pixels = 0.0;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.z + blockIdx.z * blockDim.z;

	int tid = row * pIn.step + col;

	//	if (tid < Np && blockIdx.x < gridDim.x ){
	if (blockIdx.x < gridDim.x) {
		if (pIn.data[tid] == 255 && ppDb[blockIdx.x].data[tid] == 255) {
			ppLc[blockIdx.x].data[tid] = 255;
			atomicAdd(&d_lc_pix[blockIdx.x], 1);
		}
		else {
			ppLc[blockIdx.x].data[tid] = 0;
		}
	}

	__syncthreads();  // 冒頭の #define __CUDACC__ にりより対処できている．
}

void launchMyKernel(cuda::GpuMat d_in, cuda::GpuMat *d_dbV, cuda::GpuMat *d_lcV, const int in_pixel, int *db_nzp_V, MResult *mr, double allscores[]) {

	cuda::GpuMat d_mmlc = d_dbV[0].clone();
	mr->bmid = 0;
	mr->bm_pixel_ratio = 0.0;
	int lc_pixels = 0;
	double pixel_ratio = 0;

	// __global__ 内では cv::cuda::GpuMat::ptr が使用できないため，型を変換
	cudev::GlobPtrSz<uchar> pIn = cudev::globPtr(d_in.ptr<uchar>(), d_in.step, d_in.rows, d_in.cols * d_in.channels());
	thrust::device_vector<cudev::GlobPtrSz<uchar>> pDb;
	thrust::device_vector<cudev::GlobPtrSz<uchar>> pLc;
	for (int i = 0; i < DATABASE_SIZE; i++) {
		pDb.push_back(cudev::globPtr(d_dbV[i].ptr<uchar>(), d_dbV[i].step, d_dbV[i].rows, d_dbV[i].cols * d_dbV[i].channels()));
		pLc.push_back(cudev::globPtr(d_lcV[i].ptr<uchar>(), d_lcV[i].step, d_lcV[i].rows, d_lcV[i].cols * d_lcV[i].channels()));
	}
	// device kernel 関数のなかでは vector は使えないため，生pointer で vector を表現しなおす．
	cudev::GlobPtrSz<uchar> * ppDb = thrust::raw_pointer_cast(pDb.data());
	cudev::GlobPtrSz<uchar> * ppLc = thrust::raw_pointer_cast(pLc.data());

	double *p_ratio;
	cudaMalloc((void **)&p_ratio, sizeof(double)*DATABASE_SIZE);

	unsigned int *d_lc_pix;
	unsigned int *d_db_pix;
	cudaMalloc((void **)&d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE);
	cudaMalloc((void **)&d_db_pix, sizeof(unsigned int)*DATABASE_SIZE);

	unsigned int *h_lc_pix;
	unsigned int *h_db_pix;
	h_lc_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);
	h_db_pix = (unsigned int *)malloc(sizeof(unsigned int)*DATABASE_SIZE);

	for (int i = 0; i < DATABASE_SIZE; i++) {
		h_lc_pix[i] = 0;
		h_db_pix[i] = 0;
	}

	cudaMemcpy(d_lc_pix, h_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyHostToDevice);

	// GPUの設定
	const dim3 block(1, 32, 32);
	const dim3 grid(DBSIZE, cudev::divUp(d_in.rows, block.y), cudev::divUp(d_in.cols, block.z));

	// kernel 呼び出し
	bitwise_and << <grid, block >> > (pIn, ppDb, ppLc, d_db_pix, d_lc_pix, in_pixel);
	cudaThreadSynchronize();

	cudaMemcpy(h_lc_pix, d_lc_pix, sizeof(unsigned int)*DATABASE_SIZE, cudaMemcpyDeviceToHost);

	for (int i = 0; i < DATABASE_SIZE; i++) {
		//lc_pixels = cuda::countNonZero(d_lcV[i]);
		//cout << h_pix[i] << ", " << lc_pixels << endl;
		pixel_ratio = (double)h_lc_pix[i] / (double)in_pixel;



		allscores[i] = pixel_ratio;
		//result_log[i] = pixel_ratio;
		if (mr->bm_pixel_ratio < pixel_ratio) {
			mr->bm_pixel_ratio = pixel_ratio;
			mr->bmid = i;
			d_mmlc = d_lcV[i].clone();
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

	//// DB image file の読み込み //////////////////////////////////////////////////
	auto STime = chrono::system_clock::now();
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


	for (int i = 0; i < db_size; i++) {
		temp = imread(DBpath + "/" + dbfiles[i], 0);
		temp = ~temp;
		dbV.push_back(temp);
		db_nzp_V[i] = countNonZero(temp); /////////////////////////////
	}
	printDBpropaty(dbV, dbfiles, db_size);

	auto MTime1 = chrono::system_clock::now();
	auto dur1 = MTime1 - STime;
	auto msec1 = chrono::duration_cast<chrono::milliseconds>(dur1).count();
	cout << " Reading img files: " << msec1 << " [ms]" << endl;


	//// DB image を divece メモリへ転送，LC image の領域を確保 //////////////////
	Mat Black_img = Mat::zeros(dbV[0].rows, dbV[0].cols, CV_8UC1);
	Mat dummy_img, dummy2;

	//deviceメモリの確保
	cuda::GpuMat d_img(Black_img), d_in(Black_img), d_db(Black_img), d_lc(Black_img), d_mmlc(Black_img);
	cuda::GpuMat d_dummy1, d_dummy2;

	cuda::GpuMat d_dbV[DATABASE_SIZE];
	cuda::GpuMat d_lcV[DATABASE_SIZE];

	for (int i = 0; i < dbV.size(); i++) {
		d_dummy1.upload(dbV[i]);		// このような手続きでないと，device メモリ上に実体がコピーされない． 
		d_dbV[i] = d_dummy1.clone();	    //
		d_dummy2.upload(Black_img);		// このような手続きでないと，device メモリ上に実体がコピーされない． 
		d_lcV[i] = d_dummy2.clone();	    //
	}
	//////////////////////////////////////////////////////////////////////////////
	auto int2 = chrono::system_clock::now();
	auto dur2 = int2 - MTime1;
	auto msec2 = chrono::duration_cast<chrono::milliseconds>(dur2).count();
	cout << "Copy to CUDA memory: " << msec2 << " [ms]" << endl;

	string str2;
	cout << "    SEN directory: ";
	cin >> str2;
	if (str2 != "0") inputpath = str2;

	Mat src(Black_img);
	while (1) {
		///// input 画像の前処理 ///////////////////////////////////////////////////////		
		cout << endl << "     Input a file name: ";
		cin >> infile;

		auto int3 = chrono::system_clock::now();

		src = Black_img.clone();
		src = cvMatBinary(inputpath + "/" + infile);

		cout << " size[]: " << src.size().width << "," << src.size().height << endl;

		// resize
		double ratio = dbV[0].cols / (double)src.cols;
		Mat rsrc = Mat::ones(src.rows * ratio, src.cols * ratio, CV_8UC1);
		resize(src, rsrc, rsrc.size(), INTER_LINEAR);
		rsrc = ~rsrc;

		cout << " size[]: " << rsrc.size().width << "," << rsrc.size().height << endl;
		imshow("in", rsrc);
		Mat aaa = dbV[1];
		imshow("db", aaa);
		// ノイズ除去 0 or 255 に分類 !!!必須!!!
		Mat bisrc;
		threshold(rsrc, bisrc, 120, 255, THRESH_BINARY);

		int src_pixel = countNonZero(bisrc);

		// 線分を膨張させる
		Mat dilated;
		dilate(bisrc, dilated, Mat(), Point(-1, -1), DILATE_ITERATIONS);
		d_in.upload(dilated);
		int d_dil_pixel = cuda::countNonZero(d_in);
		////////////////////////////////////////////////////////////////////////////////////

		cout << "call kernel" << endl;

		//// 論理積演算 プロセス /////////////////////////////////////////////////////////////
		MResult *mr;
		mr = (MResult *)malloc(sizeof(MResult));
		memset(mr, 0, sizeof(MResult));
		double all_score_out[DATABASE_SIZE];
		launchMyKernel(d_in, d_dbV, d_lcV, src_pixel, db_nzp_V, mr, all_score_out);


		for (int ii = 0; ii < db_size; ii++) {
			string dbimgnum = dbfiles[ii].substr(0, 8);
			string x_str = dbfiles[ii].substr(9, 8);
			string y_str = dbfiles[ii].substr(18, 8);
			//string z_str = dbfiles[ii].substr(27, 8);
			string th_str = dbfiles[ii].substr(37, 6);
			string name = dbfiles[ii].substr(0, 51);
			ofstream outputfile("C:\\Users\\tmem\\Desktop\\kawabe\\2019\\pix_count_mathing\\resultcsv\\" + infile + ",result.csv", ios::app);
			outputfile << dbimgnum << "," << x_str << "," << y_str << "," << th_str << "," << all_score_out[ii] << "," << endl;
			outputfile.close();
		}








		Mat h_bmlcimg;
		d_lcV[mr->bmid].download(h_bmlcimg);

		///// DBtable を参照して、 X Y Z を抽出　////////// 
		//double id, th, ph, ex, ey, ez, cx, cy, cz;

		//string str = DBtableLookUp(dbfiles[mr.bmid], DBtable);
		//cout << str << endl;
		//sscanf(str.data(), "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", id, th, ph, ex, ey, ez, cx, cy, cz);
		//sscanf(str.data(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &id, &th, &ph, &ex, &ey, &ez, &cx, &cy, &cz);

		//cout << "x=" << ex << " y=" << ey << " z=" << ez << " th=" << th << " ph=" << ph << endl;

		cout << "Sensing query:         " << infile << endl;
		cout << "Best matched DB image: " << dbfiles[mr->bmid] << endl;

		////////////////////////////////////////////////////////////////////////////////////
		auto int4 = chrono::system_clock::now();
		auto dur3 = int4 - int3;
		auto msec3 = chrono::duration_cast<chrono::milliseconds>(dur3).count();
		cout << "Time for matching: " << msec3 << " [ms]" << endl;

		///// 結果画像の表示 ///////////////////////////////////////////////////////////////
		Mat dst = Mat::ones(dbV[0].rows, dbV[0].cols, CV_8UC3);
		//Mat dst;

		vector<Mat> integrated;

		// 画像の統合	
		//integrated.clear();
		integrated.push_back(bisrc);	     // 青
		integrated.push_back(dbV[mr->bmid]); // 緑
		integrated.push_back(h_bmlcimg);	 // 白(赤)
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

		imshow("result", dst);
		string inimgname = infile.substr(0, 15);
		string dbimgname = dbfiles[mr->bmid].substr(0, 51);
		imwrite("C:\\Users\\tmem\\Desktop\\kawabe\\2019\\pix_count_mathing\\resultimg\\in_" + infile +",db_"+ dbfiles[mr->bmid] +".png", dst);
		waitKey(0);

		dst.release();
		for (int i = 0; i < integrated.size(); i++) integrated[i].release();
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
	if (ifs.fail()) {
		std::cerr << "DB table open に失敗" << std::endl;
		return "-1";
	}
	while (getline(ifs, str)) {
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
	Mat element = cv::getStructuringElement(MORPH_RECT, Size(2 * DILATE_ITERATIONS + 1, 2 * DILATE_ITERATIONS + 1));
	cuda::GpuMat d_element(element);

	Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_src.type(), element);
	dilateFilter->apply(d_src, d_img);
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
