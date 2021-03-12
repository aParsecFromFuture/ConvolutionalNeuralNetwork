#include "elu.h"

Elu::Elu() {

}

void Elu::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);
}

void Elu::feedforward(const float* inp, float* out) {
	for (int i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : (pow(EULER, inp[i]) - 1.0f);
}

void Elu::backpropagation(const float* inp, const float* target, float* out) {
	for (int i = 0; i < (ocrow * occol * odepth * cbatch); i++)
		out[i] = (out[i] > 0.0f) ? inp[i] : (inp[i] * pow(EULER, out[i]));
}

void Elu::test(const float* inp, float* out, int csample) {
	for (int i = 0; i < (icrow * iccol * idepth * csample); i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : (pow(EULER, inp[i]) - 1.0f);
}

void Elu::save_to(std::ofstream& file) {
	file.write((char*)&Elu::SERIALIZE_ID, sizeof(int));
	Layer::save_to(file);
}

void Elu::load_from(std::ifstream& file) {
	Layer::load_from(file);
}
