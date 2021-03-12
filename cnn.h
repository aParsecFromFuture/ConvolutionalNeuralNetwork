#pragma once
#include <fstream>
#include "layer.h"
#include "dense.h"
#include "conv2d.h"
#include "batchnorm.h"
#include "maxpool.h"
#include "relu.h"
#include "leaky_relu.h"
#include "tanh.h"
#include "sigmoid.h"
#include "softmax.h"
#include "image_array.h"
#include "label_array.h"

class CNN {
private:
	int layer_count;
	int batch_size;
	int category_count;
	Layer** layer;
	static float evaluate(const float*, const float*, int, int);
public:
	CNN();
	~CNN();
	void add_layer(Layer*);
	void setup(int, int, int, int, int);
	void train(const ImageArray&, const LabelArray&, int = 1, float = 0.1f, float = 0.1f, float = 0.0f);
	LabelArray test(const ImageArray&);
	void save_to(const char*);
	void load_from(const char*);
};