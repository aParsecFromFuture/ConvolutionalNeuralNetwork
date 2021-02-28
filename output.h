#pragma once
#include "layer.h"

class Output : public Layer{
private:
	float* weight, * mweight, * copy_weight;
	float* bias, * mbias;
private:
	Output() = default;
public:
	static const int SERIALIZE_ID = 6;
public:
	Output(int);
	void init(int, int, int, int);
	void feedforward(const float*, float*);
	void backpropagation(const float*, const float*, float*);
	void test(const float*, float*);
	void save_to(std::ofstream&);
	void load_from(std::ifstream&);
	friend class CNN;
};