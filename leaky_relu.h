#pragma once
#include "layer.h"

class LeakyRelu : public Layer {
public:
	static const int SERIALIZE_ID = 8;
	LeakyRelu();
	void init(int, int, int, int);
	void feedforward(const float*, float*);
	void backpropagation(const float*, const float*, float*);
	void test(const float*, float*, int);
	void save_to(std::ofstream&);
	void load_from(std::ifstream&);
	friend class ANN;
};
