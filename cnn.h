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
private:
	int layer_count;
	int batch_size;
	int category_count;
	Layer** layer;
	float** output;
	static float evaluate(const float*, const float*, int, int);
public:
	CNN();
	~CNN();
	void add_layer(Layer*);
	void setup(int, int, int, int, int);
	void train(const ImageArray&, const LabelArray&, int, float, float);
	LabelArray test(const ImageArray&);
	void save_to(const char*);
	void load_from(const char*);
};