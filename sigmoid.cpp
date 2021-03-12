#include "sigmoid.h"

Sigmoid::Sigmoid() {

}

void Sigmoid::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);
}

void Sigmoid::feedforward(const float* inp, float* out) {
	for (int i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = 1.0f / (1.0f + pow(EULER, -inp[i]));
}

void Sigmoid::backpropagation(const float* inp, const float* target, float* out) {
	float sigmoid;

	for (int i = 0; i < (ocrow * occol * odepth * cbatch); i++) {
		sigmoid = 1.0f / (1.0f + pow(EULER, -out[i]));
		out[i] = inp[i] * sigmoid * (1.0f - sigmoid);
	}
}

void Sigmoid::test(const float* inp, float* out, int csample) {
	for (int i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = 1.0f / (1.0f + pow(EULER, -inp[i]));
}

void Sigmoid::save_to(std::ofstream& file) {
	file.write((char*)&Sigmoid::SERIALIZE_ID, sizeof(int));
	Layer::save_to(file);
}

void Sigmoid::load_from(std::ifstream& file) {
	Layer::load_from(file);
}
