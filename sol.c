#include "stdio.h"

//사진 이미지들의 경로 및 저장할 이미지 경로
#define IN_BARBARA_256 "./img/Barbara_256x256_yuv400_8bit.raw"
#define IN_COUPLE_256 "./img/Couple_256x256_yuv400_8bit.raw"
#define IN_LENA_256 "./img/Lena_256x256_yuv400_8bit.raw"
#define OUT_BARBARA_512 "./img/Barbara_512x512_yuv400_8bit.raw"
#define OUT_COUPLE_512 "./img/Couple_512x512_yuv400_8bit.raw"
#define OUT_LENA_512 "./img/Lena_512x512_yuv400_8bit.raw"

//이미지를 불러올 타입 선언
typedef unsigned char PIXEL;
typedef PIXEL img256[256][256];
typedef PIXEL img512[512][512];

//이미지 메모리로 불러오기 및 저장 함수
void get256(img256 img, char * path);
void get512(img512 img, char * path);
void put256(img256 img);
void put512(img512 img);
void save512(img512 img, char * path);

//활동성, 방향성 MAP
typedef int map256[256][256];

//7x7 필터, DX = X각도
typedef int FILTER_7x7[7][7];
//Degree 0, Vertical
FILTER_7x7 D0[7][7]={
    {1,2,2,0,-2,-2,-1},
    {1,2,3,0,-3,-2,-1},
    {1,3,4,0,-4,-3,-1},
    {1,4,5,0,-5,-4,-1},
    {1,3,4,0,-4,-3,-1},
    {1,2,3,0,-3,-2,-1},
    {1,2,2,0,-2,-2,-1},
};
//Degree 45
FILTER_7x7 D45[7][7]={
    {0,3,2,1,1,1,1},
    {-3,0,4,3,2,1,1},
    {-2,-4,0,5,4,2,1},
    {-1,-3,-5,0,5,3,1},
    {-1,-2,-4,-5,0,4,2},
    {-1,-1,-2,-3,-4,0,3},
    {-1,-1,-1,-1,-2,-3,0},
};
//Degree 90, Horizontal
FILTER_7x7 D90[7][7]={
    {-1,-1,-1,-1,-1,-1,-1},
    {-2,-2,-3,-4,-3,-2,-2},
    {-2,-3,-4,-5,-4,-3,-2},
    {0,0,0,0,0,0,0},
    {2,3,4,5,4,3,2},
    {2,2,3,4,3,2,2},
    {1,1,1,1,1,1,1},
};
//Degree 145
FILTER_7x7 D145[7][7]={
    {-1,-1,-1,-1,-2,-3,0},
    {-1,-1,-2,-3,-4,0,3},
    {-1,-2,-4,-5,0,4,2},
    {-1,-3,-5,0,5,3,1},
    {-2,-4,0,5,4,2,1},
    {-3,0,4,3,2,1,1},
    {0,3,2,1,1,1,1}
};

//평가 함수
void RMSEandPSNR(img512 input, img512 output);

int main(void){
    img256 BARBARA;
    img256 COUPLE;
    img256 LENA;
    get256(BARBARA, IN_BARBARA_256);
    get256(COUPLE, IN_COUPLE_256);
    get256(LENA, IN_LENA_256);


    return 0;
}



void get256(img256 img, char * path){
    FILE*fp = fopen(path, "r");
    if (fp) {
        for(int r = 0; r < 128; r++){
            for(int c = 0; c < 128; c++)
                img[r][c] = fgetc(fp);
        }
    }
    fclose(fp);
}
void get512(img512 img, char * path){
    FILE*fp = fopen(path, "r");
    if (fp) {
        for(int r = 0; r < 512; r++){
            for(int c = 0; c < 512; c++)
                img[r][c] = fgetc(fp);
        }
    }
    fclose(fp);
}
void put256(img256 img){
    for(int r = 0; r < 128; r++){
        for(int c = 0; c < 128; c++)
            printf("%d ", img[r][c]);
        printf("\n");
    }
}
void put512(img512 img){
    for(int r = 0; r < 512; r++){
        for(int c = 0; c < 512; c++)
            printf("%d ", img[r][c]);
        printf("\n");
    }
}

void save512(img512 img, char * path){
    FILE*fp = fopen(path, "wb");
    if (fp) {
        for(int r = 0; r < 512; r++){
            for(int c = 0; c < 512; c++)
                 fputc(img[r][c], fp);
        }
    }
    fclose(fp);
}


void RMSEandPSNR(img512 input, img512 output) {
	int mn = 512 * 512;
	int sum = 0;
    int error = 0;
	
	for (int i = 0; i < 512; i++){
		for (int j = 0; j < 512; j++){
			error = input[i][j] - output[i][j];
			sum += error * error;
		}
	}
	double mse = (double)(sum / mn);
	double rmse = sqrt(mse);
	double psnr = 20 * log10(255 / rmse);

    printf("Interpolation\n");
    printf("RMSE: about %lf\n", rmse);
    printf("PSNR: about %lf\n", psnr);
    printf("\n\n");

}