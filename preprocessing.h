#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern const float EULER;
extern const float RAND_DIV;
extern const float EPSILON;

float* init_mem(int);
void copy_mem(const float*, float*, int, int, int, int*);
void print_mem(const float*, int, int, int, int);
float* create_target(int, int, int*);
float* read_data(const char*);
int* read_idata(const char*);
float* batch_normalization_train(float*, int, int, float*, float*);
float* batch_normalization_test(float*, int, int, float*, float*);
int* create_shuffle_index(int);
void shuffle(int*, int);