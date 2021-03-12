#include "dense.h"

Dense::Dense(int out_nc) {
	this->ocrow = 1;
	this->occol = 1;
	this->odepth = out_nc;

	this->weight = 0;
	this->mweight = 0;
	this->copy_weight = 0;
	this->bias = 0;
	this->mbias = 0;
}

void Dense::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(1, 1, icrow * iccol * idepth, this->ocrow, this->occol, this->odepth, batch_size);

	this->weight = init_mem(this->odepth * this->idepth);
	this->mweight = init_mem(this->odepth * this->idepth);
	this->copy_weight = init_mem(this->odepth * this->idepth);
	this->bias = init_mem(this->odepth);
	this->mbias = init_mem(this->odepth);

	for (int i = 0; i < (this->odepth * this->idepth); i++)
		this->mweight[i] = 0.0f;

	for (int i = 0; i < this->odepth; i++)
		this->mbias[i] = 0.0f;
}

void Dense::feedforward(const float* inp, float* out) {
	int i, j, k;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < odepth; j++) {
			out[i * odepth + j] = 0.0f;
			for (k = 0; k < idepth; k++)
				out[i * odepth + j] += weight[j * idepth + k] * inp[i * idepth + k];
			out[i * odepth + j] += bias[j];
		}
}

void Dense::backpropagation(const float* inp, const float* target, float* out) {
	int i, j, k;
	float sum;

	for (i = 0; i < (odepth * idepth); i++)
		copy_weight[i] = weight[i];
	
	for (i = 0; i < odepth; i++) {
		for (j = 0; j < idepth; j++) {
			sum = 0.0f;
			for (k = 0; k < cbatch; k++)
				sum += inp[k * odepth + i] * out[k * idepth + j];
			mweight[i * idepth + j] = momentum * mweight[i * idepth + j] + (1.0f - momentum) * (sum / cbatch);
			weight[i * idepth + j] -= lr * mweight[i * idepth + j];
		}
		sum = 0.0f;
		for (j = 0; j < cbatch; j++)
			sum += inp[j * odepth + i];
		mbias[i] = momentum * mbias[i] + (1.0f - momentum) * (sum / cbatch);
		bias[i] -= lr * mbias[i];
	}

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < idepth; j++) {
			out[i * idepth + j] = 0.0f;
			for (k = 0; k < odepth; k++)
				out[i * idepth + j] += copy_weight[k * idepth + j] * inp[i * odepth + k];
		}
}

void Dense::test(const float* inp, float* out, int csample) {
	int i, j, k;

	for (i = 0; i < csample; i++)
		for (j = 0; j < odepth; j++) {
			out[i * odepth + j] = 0.0f;
			for (k = 0; k < idepth; k++)
				out[i * odepth + j] += weight[j * idepth + k] * inp[i * idepth + k];
			out[i * odepth + j] += bias[j];
		}
}

void Dense::save_to(std::ofstream &file) {
	file.write((char*)&Dense::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);

	file.write((char*)weight, sizeof(float) * (odepth * idepth));
	file.write((char*)mweight, sizeof(float) * (odepth * idepth));
	file.write((char*)copy_weight, sizeof(float) * (odepth * idepth));

	file.write((char*)bias, sizeof(float) * odepth);
	file.write((char*)mbias, sizeof(float) * odepth);
}

void Dense::load_from(std::ifstream& file) {
	Layer::load_from(file);

	this->weight = init_mem(odepth * idepth);
	this->mweight = init_mem(odepth * idepth);
	this->copy_weight = init_mem(odepth * idepth);
	this->bias = init_mem(odepth);
	this->mbias = init_mem(odepth);

	file.read((char*)weight, sizeof(float) * (odepth * idepth));
	file.read((char*)mweight, sizeof(float) * (odepth * idepth));
	file.read((char*)copy_weight, sizeof(float) * (odepth * idepth));

	file.read((char*)bias, sizeof(float) * odepth);
	file.read((char*)mbias, sizeof(float) * odepth);
}
