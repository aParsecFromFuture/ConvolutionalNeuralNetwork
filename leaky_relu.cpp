#include "leaky_relu.h"

LeakyRelu::LeakyRelu() {

}

void LeakyRelu::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);
}

void LeakyRelu::feedforward(const float* inp, float* out) {
	for (int i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : (0.01f * inp[i]);
}

void LeakyRelu::backpropagation(const float* inp, const float* target, float* out) {
	for (int i = 0; i < (ocrow * occol * odepth * cbatch); i++)
		out[i] = (out[i] > 0.0f) ? inp[i] : (0.01f * inp[i]);
}

void LeakyRelu::test(const float* inp, float* out, int csample) {
	for (int i = 0; i < (icrow * iccol * idepth * csample); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : (0.01f * inp[i]);
}

void LeakyRelu::save_to(std::ofstream& file) {
	file.write((char*)&LeakyRelu::SERIALIZE_ID, sizeof(int));
	Layer::save_to(file);
}

void LeakyRelu::load_from(std::ifstream& file) {
	Layer::load_from(file);
}
