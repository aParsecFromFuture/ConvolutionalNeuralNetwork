#pragma once
#include <fstream>
#include "layer.h"
#include "dense.h"
#include "conv2d.h"
#include "batchnorm.h"
#include "maxpool.h"
#include "relu.h"
#include "output.h"
#include "image_array.h"
#include "label_array.h"

class CNN {
public:
	int layer_count;
	int batch_size;
	Layer** layer;
	float** output;
	static float evaluate(const float*, const float*, int, int);
public:
	CNN();
	~CNN();
	void add_layer(Layer*);
	void setup(int, int, int, int);
	void train(const ImageArray&, const LabelArray&, int, float, float);
	void test(const ImageArray&, const LabelArray&);
	void save_to(const char*);
	void load_from(const char*);
};