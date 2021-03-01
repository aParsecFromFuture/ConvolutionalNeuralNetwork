#pragma once
#include <fstream>
#include "preprocessing.h"

extern const float EULER;
extern const float EPSILON;

class Layer {
protected:
	int icrow, iccol, idepth;
	int icr, idcr;
	int ocrow, occol, odepth;
	int ocr, odcr;
	int cbatch;
	float lr, momentum;
public:
	Layer();
	void init(int, int, int, int, int, int, int);
	virtual void init(int, int, int, int) = 0;
	virtual void feedforward(const float*, float*) = 0;
	virtual void backpropagation(const float*, const float*, float*) = 0;
	virtual void test(const float*, float*) = 0;
	virtual void save_to(std::ofstream&);
	virtual void load_from(std::ifstream&);
	friend class CNN;
};