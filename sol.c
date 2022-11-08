#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifndef max
#define max(a,b)  (((a) > (b)) ? (a) : (b))
#endif

//--사진 이미지들의 경로 및 저장할 이미지 경로--//
//Input 이미지 256x256 경로
#define IN_BARBARA_256 "./img/Barbara_256x256_yuv400_8bit.raw"
#define IN_COUPLE_256 "./img/Couple_256x256_yuv400_8bit.raw"
#define IN_LENA_256 "./img/Lena_256x256_yuv400_8bit.raw"
//원본 이미지 512x512 경로
#define IN_BARBARA_512_ORIGINAL "./img/Barbara_512x512_yuv400_8bit_original.raw"
#define IN_COUPLE_512_ORIGINAL "./img/Couple_512x512_yuv400_8bit_original.raw"
#define IN_LENA_512_ORIGINAL "./img/Lena_512x512_yuv400_8bit_original.raw"
//결과 출력 이미지 경로
#define OUT_BARBARA_512 "./out/Barbara_512x512_yuv400_8bit.raw"
#define OUT_COUPLE_512 "./out/Couple_512x512_yuv400_8bit.raw"
#define OUT_LENA_512 "./out/Lena_512x512_yuv400_8bit.raw"

//--이미지를 불러올 타입 선언--//
typedef unsigned char PIXEL;
typedef PIXEL img256[256][256];
typedef PIXEL img256_p3[262][262]; //상하좌우 3 padding (neareast interpolation)이미지 타입
typedef PIXEL img512[512][512];

//--이미지 메모리로 불러오기 및 저장 함수--//
void get256(img256 img, char * path);
void get256_p3(img256 img, img256_p3 p3);
void get512(img512 img, char * path);
void put256(img256 img);
void put512(img512 img);
void save512(img512 img, char * path);
void save256_p3(img256_p3 img, char * path);

//--필터--//
//7x7 필터, DX = X각도
typedef int FILTER_7x7[7][7];
//Degree 0, Vertical
FILTER_7x7 D0={
    {1,2,2,0,-2,-2,-1},
    {1,2,3,0,-3,-2,-1},
    {1,3,4,0,-4,-3,-1},
    {1,4,5,0,-5,-4,-1},
    {1,3,4,0,-4,-3,-1},
    {1,2,3,0,-3,-2,-1},
    {1,2,2,0,-2,-2,-1},
};
//Degree 45
FILTER_7x7 D45={
    {-1,-1,-1,-1,-2,-3,0},
    {-1,-1,-2,-3,-4,0,3},
    {-1,-2,-4,-5,0,4,2},
    {-1,-3,-5,0,5,3,1},
    {-2,-4,0,5,4,2,1},
    {-3,0,4,3,2,1,1},
    {0,3,2,1,1,1,1}
};
//Degree 90, Horizontal
FILTER_7x7 D90={
    {-1,-1,-1,-1,-1,-1,-1},
    {-2,-2,-3,-4,-3,-2,-2},
    {-2,-3,-4,-5,-4,-3,-2},
    {0,0,0,0,0,0,0},
    {2,3,4,5,4,3,2},
    {2,2,3,4,3,2,2},
    {1,1,1,1,1,1,1},
};
//Degree 135
FILTER_7x7 D135={
    {0,3,2,1,1,1,1},
    {-3,0,4,3,2,1,1},
    {-2,-4,0,5,4,2,1},
    {-1,-3,-5,0,5,3,1},
    {-1,-2,-4,-5,0,4,2},
    {-1,-1,-2,-3,-4,0,3},
    {-1,-1,-1,-1,-2,-3,0},
};

//-- Pixel Classification 유틸 함수 --//
//7x7 행렬에 필터를 적용하는 함수
void matrix_mul_7x7(unsigned int y, unsigned int x, img256_p3 input, FILTER_7x7 filter, FILTER_7x7 matrix);
void matrix_product_7x7(unsigned int y, unsigned int x, img256_p3 input, FILTER_7x7 filter, FILTER_7x7 matrix);

//NxN 행렬식(Determinant)을 구하는 함수
//https://nate9389.tistory.com/63
int matrix_determinent(int n, int (*matrix)[n]);
int matrix_sum(int n, int (*matrix)[n]);

//Acitivity 0~4, Directive 0~4, Group 0~24
int get_activity(int D0_SUM, int D90_SUM, double denominator);
int get_directive(int D0_SUM, int D45_SUM, int D90_SUM, int D135_SUM);
int get_group(int activity, int directive);

//--Pixel Classification Data Type--//
//Group 0~24에 해당하는 Linked List
//Group은 Linked List의 길이 정보 포함
typedef struct Group{
    int count;
	struct Node * next;
	struct Node * last;
}Group;
typedef struct Node{
	int r;
    int c;
	struct Node * next;
}Node;

//길이 정보 = 0
void initGroupList(Group group[], int n);
void addList(Group * group, int r, int c);
void freeGroup(Group group[], int n);

//--핵심함수 : PixelClassification -> AdaptiveInterpolation--//
void PixelClassification(Group groupList[25], img256_p3 img256, double denominator);
void AdaptiveInterpolation(Group groupList[25], img256_p3 input, img512 original, img512 output);
void Interpolate(char * input256_path, char * origin512_path, char * output512_path, double denominator);

//--2D, 1D 동적배열 할당, 해제 함수--//
int* malloc1D(int length);
int** malloc2D(int row, int column);
void free1D(int * arr);
void free2D(int ** arr, int row);

//--AdaptiveInterpolation 유틸 함수--//
//해당 group에 있는 픽셀좌표를 중심으로 7x7 인접행렬들을 x, xT에 나열하는 함수
void adjacencyMatrix(int ** xT, int ** x, Group * group, img256_p3 img256);
//해당 group에 있는 픽셀좌표에 해당하는 원본 이미지의 h 정답, v 정답, d 정답에 대한 배열을 초기화
void getYofOriginal(int * hY, int * vY, int * dY, Group * group, img512 img512);
//NxN 역행렬
int invMatrix(int n, const double* A, double* b);

//평가 함수
void RMSEandPSNR(char* original_path, char* output_path);

int main(void){
    Interpolate(IN_BARBARA_256, IN_BARBARA_512_ORIGINAL, OUT_BARBARA_512, 2910.0);
    Interpolate(IN_COUPLE_256, IN_COUPLE_512_ORIGINAL, OUT_COUPLE_512, 2300.0);
    Interpolate(IN_LENA_256, IN_LENA_512_ORIGINAL, OUT_LENA_512, 2000.0);

    RMSEandPSNR(IN_BARBARA_512_ORIGINAL, OUT_BARBARA_512);
    RMSEandPSNR(IN_COUPLE_512_ORIGINAL, OUT_COUPLE_512);
    RMSEandPSNR(IN_LENA_512_ORIGINAL, OUT_LENA_512);

    return 0;
}


//--이미지 메모리로 불러오기 및 저장 함수--//
void get256(img256 img, char * path){
    FILE*fp = fopen(path, "r");
    if (fp) {
        for(int r = 0; r < 256; r++){
            for(int c = 0; c < 256; c++)
                img[r][c] = fgetc(fp);
        }
    }
    fclose(fp);
}
void get256_p3(img256 img, img256_p3 p3){
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 3; j++) {
			p3[j][i + 3] = img[0][i]; // top neareast copy
			p3[261 - j][i + 3] = img[255][i]; // bottom neareast copy
		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			p3[j + 3][i + 3] = img[j][i]; //copy
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 262; j++) {
			p3[j][i] = p3[j][3]; // left neareast copy
			p3[j][261 - i] = p3[j][258]; // right neareast copy
		}
	}
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
    for(int r = 0; r < 256; r++){
        for(int c = 0; c < 256; c++)
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
void save256_p3(img256_p3 img, char * path){
    FILE*fp = fopen(path, "wb");
    if (fp) {
        for(int r = 0; r < 262; r++){
            for(int c = 0; c < 262; c++)
                 fputc(img[r][c], fp);
        }
    }
    fclose(fp);
}


//-- Pixel Classification 유틸 함수 --//
//필터 적용 함수
void matrix_mul_7x7(unsigned int y, unsigned int x, img256_p3 input, FILTER_7x7 filter, FILTER_7x7 matrix){
    for(int r = 0; r<7; r++){
        for(int c = 0; c<7; c++)
            matrix[r][c] = (input[y-3+r][x-3+c] * filter[r][c]);
    }
}
void matrix_product_7x7(unsigned int y, unsigned int x, img256_p3 input, FILTER_7x7 filter, FILTER_7x7 matrix){
    for (int r = 0; r < 7; r++)
	{
		for (int c = 0; c < 7; c++)
		{
			matrix[r][c] = 0;
			for (int k = 0; k < 3; k++)
				matrix[r][c] += (input[y-3+r][x-3+c] * filter[r][c]);
		}
	}
}

//행렬식 구하는 함수
//https://nate9389.tistory.com/63
int matrix_determinent(int n, int (*matrix)[n]){
    if(n == 1) return matrix[0][0];
    int i, j, k;
	int minor_matrix[n][n-1][n-1];
	for(k = 0; k < n; k++){
		for(i = 0; i < n - 1; i++)
			for(j = 0; j < n; j++){
				if(j < k)
					minor_matrix[k][i][j] = matrix[i + 1][j];
				else if(j > k)
					minor_matrix[k][i][j - 1] = matrix[i + 1][j];
			}
	}
	int sum = 0;
	int test = 1;
	for(k = 0; k < n; k++){
		sum += test * matrix[0][k] * matrix_determinent(n - 1, minor_matrix[k]);
		test *= -1;
	}
	return sum;
}


int matrix_sum(int n, int (*matrix)[n]){
    int sum = 0;
    for(int r = 0; r < n; r++){
        for(int c = 0; c < n; c++)
            sum += matrix[r][c];
    }
	return sum;
}


//Activity 0~4, denominator를 조절할 것
int get_activity(int D0_SUM, int D90_SUM, double denominator){
    int result = (int)((D0_SUM + D90_SUM) / denominator);
    return result > 4 ? 4 : result;
}

//Directive 0~4
int get_directive(int D0_SUM, int D45_SUM, int D90_SUM, int D135_SUM){
    int max_value = max(max(D0_SUM, D45_SUM), max(D90_SUM, D135_SUM));
    double mean = (D0_SUM + D45_SUM + D90_SUM + D135_SUM - max_value) / 3;
    if (max_value <= mean * 1.5)
        return 0;
    else if(max_value == D0_SUM)
        return 1;
    else if(max_value == D45_SUM)
        return 2;
    else if(max_value == D90_SUM)
        return 3;
    else if(max_value == D135_SUM)
        return 4;
    return 0;
}

//Group 0~24, Acitvity * 5 + Directivve
int get_group(int activity, int directive){
    return activity * 5 + directive;
}

void initGroupList(Group group[], int n){
    for(int i = 0; i < n; i++)
    {
        group[i].count = 0;
        group[i].next = NULL;
        group[i].last = NULL;
    }
}
void addList(Group * group, int r, int c){
    Node* node = (Node*)malloc(sizeof(Node));
    node->r = r;
    node->c = c;
    node->next = NULL;
    if(group->last == NULL){
        group->next = node;
        group->last = node;
    }
    else{
        Node* cur = group->last;
        cur->next = node;
        group->last = node;
    }
    (group->count) += 1;
}
void freeGroup(Group group[], int n){
    for(int i = 0; i < n; i++){
        Node *cur = group[i].next;
        while (cur != NULL)
        {
            Node *next = cur->next;
            free(cur);
            cur = next;
         }
    }
}

void PixelClassification(Group groupList[25], img256_p3 img256_p3, double denominator)
{
    FILTER_7x7 D0_MUL;
    FILTER_7x7 D45_MUL;
    FILTER_7x7 D90_MUL;
    FILTER_7x7 D135_MUL;

    int D0_SUM;
    int D45_SUM;
    int D90_SUM;
    int D135_SUM;

    int activity;
    int directive;
    int group;

    for(int r = 3; r<259; r++){
        for(int c = 3; c<259; c++){
            matrix_product_7x7(r, c, img256_p3, D0, D0_MUL);
            matrix_product_7x7(r, c, img256_p3, D45, D45_MUL);
            matrix_product_7x7(r, c, img256_p3, D90, D90_MUL);
            matrix_product_7x7(r, c, img256_p3, D135, D135_MUL);
            
            D0_SUM = abs(matrix_sum(7, D0_MUL));
            D45_SUM = abs(matrix_sum(7, D45_MUL));
            D90_SUM = abs(matrix_sum(7, D90_MUL));
            D135_SUM = abs(matrix_sum(7, D135_MUL));

            activity = get_activity(D0_SUM, D90_SUM, denominator);
            directive = get_directive(D0_SUM, D45_SUM, D90_SUM, D135_SUM);
            group = get_group(activity, directive);

            addList(&groupList[group], r-3, c-3);
        }
    }
}

void AdaptiveInterpolation(Group groupList[25], img256_p3 input, img512 original, img512 output){
    for(int group = 0; group < 25; group++){
        int groupLength = groupList[group].count;
        printf("\t[%d/25] apply F to H,V,D in group and apply image, %d pixel data\n", group+1 , groupLength);

    
        int ** xT = malloc2D(49, groupLength);;
        int ** x = malloc2D(groupLength, 49);
        adjacencyMatrix(xT, x, &groupList[group], input);

        //xTx 계산 (49 * length) * (length * 49) = 49 * 49
        double xTx[49][49]={{0,},};
        for (int k = 0; k < 49; k++){
            for (int i = 0; i < 49; i++){
                for (int j = 0; j < groupLength; j++)
					xTx[k][i] += (xT[k][j] * x[j][i]);
            }
        }

        //xTxI 49x49
        double xTxI[49][49]={{0,},};
        invMatrix(49, (double*)xTx, (double*)xTxI);

        //y
        int * hY = malloc1D(groupLength);
        int * vY  = malloc1D(groupLength);
        int * dY  = malloc1D(groupLength);
        getYofOriginal(hY, vY, dY, &groupList[group], original);

        //xTy (49 * length) * (length * 1) = 49 * 1
        int xTyH[49] = {0,};
        int xTyV[49] = {0,};
        int xTyD[49] = {0,};
        for (int i = 0; i < 49; i++)
        {
            for (int j = 0; j < groupLength; j++) {
				xTyH[i] += (xT[i][j] * hY[j]);
				xTyV[i] += (xT[i][j] * dY[j]);
				xTyD[i] += (xT[i][j] * vY[j]);
			}
        }

        //F = xTxIxTy (49 * 49) * (49 * 1) = 49 * 1
        double hF[49]={0,};
        double vF[49]={0,};
        double dF[49]={0,};
        for (int i = 0; i < 49; i++) {
			for (int j = 0; j < 49; j++) {
				hF[i] += (xTxI[i][j] * xTyH[j]);
				vF[i] += (xTxI[i][j] * xTyV[j]);
				dF[i] += (xTxI[i][j] * xTyD[j]);
			}
		}

        //현재 Group에 존재하는 픽셀들에게 필터 적용
        Node * cur = groupList[group].next;
        int node_i = 0;
        while(cur!=NULL){
			double H_pixel = 0;
			double D_pixel = 0;
			double V_pixel = 0;
			for (int j = 0; j < 49; j++) {
				H_pixel += x[node_i][j] * hF[j];
				D_pixel += x[node_i][j] * vF[j];
				V_pixel += x[node_i][j] * dF[j];
			}
            H_pixel = (H_pixel < 0 ? 0 : H_pixel);
            H_pixel = (H_pixel > 255 ? 255 : H_pixel);
            D_pixel = (D_pixel < 0 ? 0 : D_pixel);
            D_pixel = (D_pixel > 255 ? 255 : D_pixel);
            V_pixel = (V_pixel < 0 ? 0 : V_pixel);
            V_pixel = (V_pixel > 255 ? 255 : V_pixel);
			
			output[2 * (cur->r) + 1][2 * (cur->c)] = (PIXEL)(H_pixel + 0.5); // H
			output[2 * (cur->r)][2 * (cur->c) + 1] = (PIXEL)(V_pixel + 0.5); // V
            output[2 * (cur->r)][2 * (cur->c)] = (PIXEL)(D_pixel + 0.5); // D
			output[2 * (cur->r) + 1][2 * (cur->c) + 1] = input[(cur->r)+3][(cur->c)+3]; // Original
            
            cur = cur->next;
            node_i++;
		}

        free2D(xT, 49);
        free2D(x, groupLength);
        free1D(hY);
        free1D(vY);
        free1D(dY);
    }
}

void Interpolate(char * input256_path, char * origin512_path, char * output512_path, double denominator){
    printf("----------------------------------------\n");
    printf("Interpolate %s\n", input256_path);

    //256x256 이미지를 불러옴
    img256 img256;
    get256(img256, input256_path);
    printf("[1/6] Success to read 256x256 image for interplating\n");

    //256x256 -> 262x262 로 상하좌우 3 padding
    img256_p3 img256_p3;
    get256_p3(img256, img256_p3);
    printf("[2/6] Success to padding 256x256 image to 262x262\n");

    //Group 데이터구조에 Padding된 이미지의 Pixel 정보에 대해 Classification을 진행
    Group groupList[25];
    initGroupList(groupList, 25);
    PixelClassification(groupList, img256_p3, denominator);
    printf("[3/6] Success to PixelClassification\n");

    //512x512 원본이미지를 불러옴
    img512 img512_origin;
    get512(img512_origin, origin512_path);
    printf("[4/6] Success to read 512x512 original image\n");

    //512x512 원본이미지를 통해 Pixel Classification정보에 따라 학습을 진행하고 interpolation
    img512 img512_output;
    AdaptiveInterpolation(groupList, img256_p3, img512_origin, img512_output);
    printf("[5/6] Success to Interpolate\n");

    //결과 저장
    save512(img512_output, output512_path);
    printf("[6/6] Success to save output\n\n");

    freeGroup(groupList, 25);
}

//--2D, 1D 동적배열 할당, 해제 함수--//
int * malloc1D(int length){
    int * arr = (int*) malloc ( sizeof(int) * length );
    return arr;
}
int ** malloc2D(int row, int column){
    int ** arr = (int**) malloc ( sizeof(int*) * row );
    for(int r=0; r<row; r++)
        arr[r] = (int*) malloc ( sizeof(int) * column );
    return arr;
}
void free1D(int * arr){
   free(arr);
}
void free2D(int ** arr, int row){
   for(int r=0; r<row; r++) free(arr[r]);
   free(arr);
}

//--AdaptiveInterpolation 유틸 함수--//
void adjacencyMatrix(int ** xT, int ** x, Group * group, img256_p3 img256){
    int node_i = 0;
    Node* cur = group->next;
    while (cur != NULL) {
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                int row = (cur->r) + r;
                int col = (cur->c) + c;
                xT[r * 7 + c][node_i] = img256[row][col];
                x[node_i][r * 7 + c] = img256[row][col];
            }
        }
        cur = cur -> next;
        node_i++;
    }
}
void getYofOriginal(int * hY, int * vY, int * dY, Group * group, img512 img512){
    int node_i = 0;
    Node* cur = group->next;
    while (cur != NULL) {
        *(hY+node_i) = img512[2 * (cur->r) + 1][2 * (cur->c)]; //(1, 0)
        *(vY+node_i) = img512[2 * (cur->r)][2 * (cur->c) + 1]; //(0, 1)
        *(dY+node_i) = img512[2 * (cur->r)][2 * (cur->c)]; //(0, 0)
        cur = cur -> next;
        node_i++;
    }
}

//역행렬 구하는 함수
//https://200315193.tistory.com/332
int invMatrix(int n, const double* A, double* b)  
{
    double m;
    register int i, j, k;
    double* a = (double*) malloc ( sizeof(double) * n*n );

    if(a==NULL) return 0;
    for(i=0; i<n*n; i++) 
        a[i] = A[i];
    for(i=0; i<n; i++) 
    {
        for(j=0; j<n; j++)
            b[i*n+j]=(i==j)?1.:0.;
    }
    for(i=0; i<n; i++)
    {
        if(a[i*n+i]==0.) 
        {
            if(i==n-1) 
            {
                free(a);
                return 0;
            }
            for(k=1; i+k<n; k++)
            {
                if(a[i*n+i+k] != 0.) 
                break;
            }
            if(i+k>=n)
            {
                free(a);
                return 0;
            }
            for(j=0; j<n; j++) 
            {
                m = a[i*n+j];
                a[i*n+j] = a[(i+k)*n+j];
                a[(i+k)*n+j] = m;
                m = b[i*n+j];
                b[i*n+j] = b[(i+k)*n+j];
                b[(i+k)*n+j] = m;
            }
        }
        m = a[i*n+i];
        for(j=0; j<n; j++) 
        {
            a[i*n+j]/=m;
            b[i*n+j]/=m;
        }
        for(j=0; j<n; j++) 
        {
            if(i==j) continue;

            m = a[j*n+i];
            for(k=0; k<n; k++)   
            {
                a[j*n+k] -= a[i*n+k]*m;
                b[j*n+k] -= b[i*n+k]*m;
            }
        }
    }
    free(a);
    return 1;
}



void RMSEandPSNR(char* original_path, char* output_path) {
    printf("----------------------------------------\n");
    printf("Interpolated %s\n", output_path);
    img512 original;
    get512(original, original_path);
    
    img512 output;
    get512(output, output_path);

	int mn = 512 * 512;
	int sum = 0;
    int error = 0;
	
	for (int i = 0; i < 512; i++){
		for (int j = 0; j < 512; j++){
			error = original[i][j] - output[i][j];
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