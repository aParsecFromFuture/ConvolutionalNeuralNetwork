#pragma once
#include "layer.h"

class BatchNorm : public Layer {
private:
    float* gamma, * mgamma, * dgamma, * gammax;
    float* beta, * mbeta, * dbeta;
    float* var, * dvar, * mean, * dmean;
    float* xmean, * dxmean;
    float* xhat, * dxhat;
    float* std, * dstd, * istd, * distd;
    float* xmean2, * dxmean11, * dxmean12, * dxmean2;
    float* dx1, * dx2;
public:
    static const int SERIALIZE_ID = 1;
public:
    BatchNorm();
    void init(int, int, int, int);
    void feedforward(const float*, float*);
    void backpropagation(const float*, const float*, float*);
    void test(const float*, float*);
    void save_to(std::ofstream&);
    void load_from(std::ifstream&);
    friend class CNN;
};