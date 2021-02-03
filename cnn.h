#pragma once
#include "preprocessing.h"

#define SHAPE1(x) (x)
#define SHAPE2(x, y) ((x) * (y))
#define SHAPE3(x, y, z) ((x) * (y) * (z))
#define SHAPE4(x, y, z, v) ((x) * (y) * (z) * (v))

extern int EPOCH;
extern float BATCH_DIV;
extern float LEARNING_RATE;
extern float MOMENTUM;

struct batchnorm {
    int icrow;
    int iccol;
    int idepth;
    int cbatch;
    int icr;
    int idcr;
    float* gamma;
    float* mgamma;
    float* dgamma;
    float* beta;
    float* mbeta;
    float* dbeta;
    float* mean;
    float* dmean;
    float* var;
    float* dvar;
    float* xhat;
    float* dxhat;
    float* gammax;
    float* std;
    float* dstd;
    float* istd;
    float* distd;
    float* xmean;
    float* dxmean;
    float* xmean2;
    float* dxmean11;
    float* dxmean12;
    float* dxmean2;
    float* dx1;
    float* dx2;
};

struct conv {
    int icrow;
    int iccol;
    int idepth;
    int kcrow;
    int kccol;
    int kdepth;
    int ocrow;
    int occol; 
    int cbatch;
    int icr;
    int idcr;
    int kcr;
    int kdcr;
    int ocr;
    int odcr;
    float* kernel;
    float* bias;
    float* copy_kernel;
    float* mkernel;
    float* mbias;

};

struct maxp {
    int icrow;
    int iccol;
    int idepth;
    int cbatch;
    int ocrow;
    int occol;
    int icr;
    int idcr;
    int ocr;
    int odcr;
    float* maxmap;
};

struct dense {
    int inp_nc;
    int out_nc;
    int cbatch;
    float* weight;
    float* bias;
    float* copy_weight;
    float* mweight;
    float* mbias;
};

struct batchnorm create_batchnorm(int, int, int, int);
struct conv create_conv(int, int, int, int, int, int, int);
struct maxp create_maxp(int, int, int, int);
struct dense create_dense(int, int, int);

void f_relu(const float*, int, float*);
void f_tanh(const float*, int, float*);
void f_conv(const float*, struct conv*, float*);
void f_batchnorm(const float*, struct batchnorm*, float*);
void f_maxp(const float*, struct maxp*, float*);
void f_dense(const float*, struct dense*, float*);
void f_output(const float*, struct dense*, float*);

float evaluate(const float*, const float*, int, int);

void b_relu(const float*, int, float*);
void b_tanh(const float*, int, float*);
void b_conv(const float*, struct conv*, float*);
void b_batchnorm(const float*, struct batchnorm*, float*);
void b_maxp(const float*, struct maxp*, float*);
void b_dense(const float*, struct dense*, float*);
void b_output(const float*, const float*, struct dense*, float*);
