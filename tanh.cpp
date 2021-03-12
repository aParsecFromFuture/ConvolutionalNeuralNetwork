#include "tanh.h"

Tanh::Tanh() {

}

void Tanh::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);
}

void Tanh::feedforward(const float* inp, float* out) {
	for (int i = 0; i < (icrow * iccol * idepth * cbatch); i++)
		out[i] = 2.0f / (1.0f + pow(EULER, -2.0f * inp[i])) - 1.0f;
}

void Tanh::backpropagation(const float* inp, const float* target, float* out) {
	float tanh;

	for (int i = 0; i < (ocrow * occol * odepth * cbatch); i++) {
		tanh = 2.0f / (1.0f + pow(EULER, -2.0f * inp[i])) - 1.0f;
		out[i] = inp[i] * (1.0f - tanh * tanh);
	}
}

void Tanh::test(const float* inp, float* out, int csample) {
	int i;
	float ePos, eNeg;

	for (i = 0; i < (icrow * iccol * idepth * cbatch); i++) {
		ePos = pow(EULER, inp[i]);
		eNeg = pow(EULER, -inp[i]);
		out[i] = (ePos - eNeg) / (ePos + eNeg);
	}
}

void Tanh::save_to(std::ofstream& file) {
	file.write((char*)&Tanh::SERIALIZE_ID, sizeof(int));
	Layer::save_to(file);
}

void Tanh::load_from(std::ifstream& file) {
	Layer::load_from(file);
}
