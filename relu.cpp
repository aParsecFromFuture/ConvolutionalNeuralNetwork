#include "relu.h"

Relu::Relu() {

}

void Relu::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);
}

void Relu::feedforward(const float* inp, float* out) {
	int i;

	for (i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : 0.0f;
}

void Relu::backpropagation(const float* inp, const float* target, float* out) {
	int i;

	for (i = 0; i < (ocrow * occol * odepth * cbatch); i++)
		out[i] = (out[i] > 0.0f) ? inp[i] : 0.0f;
}

void Relu::test(const float* inp, float* out, int csample) {
	int i;

	for (i = 0; i < (icrow * iccol * idepth * csample); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : 0.0f;
}

void Relu::save_to(std::ofstream &file) {
	file.write((char*)&Relu::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);
}

void Relu::load_from(std::ifstream& file) {
	Layer::load_from(file);
}
