#pragma once
#include "layer.h"

class Relu : public Layer {
public:
	static const int SERIALIZE_ID = 5;
	Relu();
	void init(int, int, int, int);
	void feedforward(const float*, float*);
	void backpropagation(const float*, const float*, float*);
	void test(const float*, float*);
	void save_to(std::ofstream&);
	void load_from(std::ifstream&);
	friend class CNN;
};