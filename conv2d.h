#pragma once
#include "layer.h"

class Conv2D : public Layer {
private:
	int kcrow, kccol, kdepth;
	int kcr, kdcr;
	float* kernel, * mkernel, * copy_kernel;
	float* bias, * mbias;
private:
	Conv2D();
public:
	static const int SERIALIZE_ID = 2;
public:
	Conv2D(int, int, int);
	void init(int, int, int, int);
	void feedforward(const float*, float*);
	void backpropagation(const float*, const float*, float*);
	void test(const float*, float*);
	void save_to(std::ofstream&);
	void load_from(std::ifstream&);
	friend class CNN;
};
