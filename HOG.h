#pragma once

#include "feature.h"
#include <vector>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>

using namespace std;

#define CELL_SIZE				8		//8*8pixel
#define BLOCK_SIZE				2		//2*2cell
#define ORIENTATION				9
#define LN_E					1.0
#define PI						(3.1415926535897932384626433832795)
#define WINDOW_SIZE_X			32
#define WINDOW_SIZE_Y			32
#define OB_CENTER_Z				-0.7
#define SET_X_SIZE				32		//低解像度にしたときのXの大きさ
#define SET_Y_SIZE				32		//低解像度にしたときのYの大きさ

//1パッチに含まれるブロック数
#define BLOCK_NUM			(SET_X_SIZE/CELL_SIZE - BLOCK_SIZE + 1)*(SET_Y_SIZE/CELL_SIZE - BLOCK_SIZE + 1)
//セルの数
#define CELL_NUM			(BLOCK_NUM * BLOCK_SIZE * BLOCK_SIZE)
//特徴量の数
#define FEATURE_NUM			(CELL_NUM * ORIENTATION)

class CHOG
{
public:	
	void Init(IplImage* img);
	void CompHist(IplImage* img);
	void Getfeature(double hog_feature[]);

private:
	int bpp;
	int wStep;
	double cell_hist[SET_X_SIZE / CELL_SIZE][SET_Y_SIZE / CELL_SIZE][9];
};
