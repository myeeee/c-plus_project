#include "HOG.h"

void CHOG::Init(IplImage* img)
{
	wStep = img->widthStep;
	bpp = ((img->depth & 255) / 8) * img->nChannels;
}

void CHOG::CompHist(IplImage* img)
{
	int x, y;
	double xgrad, ygrad, magnitude, grad;

	unsigned char* imgSource = (unsigned char*)img->imageData;

	memset(cell_hist, 0, sizeof(cell_hist));

	for (y = 0; y < SET_Y_SIZE; y++){
		for (x = 0; x < SET_X_SIZE; x++){
			if (x == 0){
				xgrad = imgSource[y*wStep + (x + 0)*bpp] - imgSource[y*wStep + (x + 1)*bpp];
			}
			else if (x == img->width - 1){
				xgrad = imgSource[y*wStep + (x - 1)*bpp] - imgSource[y * wStep + (x + 0)*bpp];
			}
			else{
				xgrad = imgSource[y * wStep + (x - 1)* bpp] - imgSource[y * wStep + (x + 1)*bpp];
			}
			if (y == 0){
				ygrad = imgSource[(y + 0)*wStep + x*bpp] - imgSource[(y + 1)*wStep + x*bpp];
			}
			else if (y == img->height - 1){
				ygrad = imgSource[(y - 1)*wStep + x*bpp] - imgSource[(y + 0)*wStep + x*bpp];
			}
			else{
				ygrad = imgSource[(y - 1)*wStep + x*bpp] - imgSource[(y + 1)*wStep + x*bpp];
			}

			magnitude = sqrt(xgrad *xgrad + ygrad *ygrad);

			grad = atan2(ygrad, xgrad);

			grad = (grad * 180) / PI;
			grad -= 1.0;

			if (grad < 0.0){
				grad += 360.0;
			}
			if (grad > 180.0){
				grad = grad - 180.0;
			}
			grad = (int)grad / 20;

			cell_hist[x / CELL_SIZE][y / CELL_SIZE][(int)grad] += magnitude;
		}
	}
}

void CHOG::Getfeature(double hog_feature[])
{
	int i, j, p, q, k;
	double sum_magnitude;
	int c = 0;
	double e = LN_E;

	for (q = 0; q < ((SET_Y_SIZE / CELL_SIZE) - BLOCK_SIZE + 1); q++){
		for (p = 0; p < ((SET_X_SIZE / CELL_SIZE) - BLOCK_SIZE + 1); p++){
			sum_magnitude = 0.0;

			for (j = 0; j < BLOCK_SIZE; j++){
				for (i = 0; i < BLOCK_SIZE; i++){
					for (k = 0; k < ORIENTATION; k++){
						sum_magnitude += cell_hist[p + i][q + j][k] * cell_hist[p + i][q + j][k];
					}
				}
			}
			sum_magnitude = 1.0 / sqrt(sum_magnitude + e);
			for (j = 0; j < BLOCK_SIZE; j++){
				for (i = 0; i < BLOCK_SIZE; i++){
					for (k = 0; k < ORIENTATION; k++){
						hog_feature[c] = cell_hist[p + i][q + j][k] * sum_magnitude;
						c++;
					}
				}
			}
		}
	}
}