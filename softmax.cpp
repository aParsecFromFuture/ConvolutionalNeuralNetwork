#include "softmax.h"

Softmax::Softmax() {
	
}

void Softmax::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(1, 1, icrow * iccol * idepth, 1, 1, icrow * iccol * idepth, batch_size);
}

void Softmax::feedforward(const float* inp, float* out) {
	int i, j;
	float sum;

	for (i = 0; i < cbatch; i++) {
		sum = 0.0f;

		for (j = 0; j < odepth; j++)
			sum += pow(EULER, inp[i * odepth + j]);

		sum = 1.0f / sum;

		for (j = 0; j < odepth; j++)
			out[i * odepth + j] = pow(EULER, inp[i * odepth + j]) * sum;
	}
}

void Softmax::backpropagation(const float* inp, const float* target, float* out) {
	int i, j;
	float sum;

	for (i = 0; i < cbatch; i++) {
		sum = 0.0f;

		for (j = 0; j < odepth; j++)
			sum += target[i * odepth + j];

		for (j = 0; j < odepth; j++)
			out[i * odepth + j] = sum * inp[i * odepth + j] - target[i * odepth + j];
	}
}

void Softmax::test(const float* inp, float* out, int csample) {
	int i, j;
	float sum;

	for (i = 0; i < csample; i++) {
		sum = 0.0f;

		for (j = 0; j < odepth; j++)
			sum += pow(EULER, inp[i * odepth + j]);

		sum = 1.0f / sum;
		
		for (j = 0; j < odepth; j++)
			out[i * odepth + j] = pow(EULER, inp[i * odepth + j]) * sum;
	}
}

void Softmax::save_to(std::ofstream& file) {
	file.write((char*)&Softmax::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);
}

void Softmax::load_from(std::ifstream& file) {
	Layer::load_from(file);
}