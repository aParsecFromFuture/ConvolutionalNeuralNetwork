#include "dense.h"

Dense::Dense(int out_nc) {
	this->ocrow = out_nc;
	this->occol = 1;
	this->odepth = 1;

	this->weight = 0;
	this->mweight = 0;
	this->copy_weight = 0;
	this->bias = 0;
	this->mbias = 0;
}

void Dense::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow * iccol * idepth, 1, 1, this->ocrow, this->occol, this->odepth, batch_size);

	this->weight = init_mem(this->ocrow * this->icrow);
	this->mweight = init_mem(this->ocrow * this->icrow);
	this->copy_weight = init_mem(this->ocrow * this->icrow);
	this->bias = init_mem(this->ocrow);
	this->mbias = init_mem(this->ocrow);

	for (int i = 0; i < (this->ocrow * this->icrow); i++)
		this->mweight[i] = 0.0f;

	for (int i = 0; i < this->ocrow; i++)
		this->mbias[i] = 0.0f;
}

void Dense::feedforward(const float* inp, float* out) {
	int i, j, k;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < ocrow; j++) {
			out[i * ocrow + j] = 0.0f;
			for (k = 0; k < icrow; k++)
				out[i * ocrow + j] += weight[j * icrow + k] * inp[i * icrow + k];
			out[i * ocrow + j] += bias[j];
		}
}

void Dense::backpropagation(const float* inp, const float* target, float* out) {
	int i, j, k;
	float sum;

	for (i = 0; i < (ocrow * icrow); i++)
		copy_weight[i] = weight[i];
	
	for (i = 0; i < ocrow; i++) {
		for (j = 0; j < icrow; j++) {
			sum = 0.0f;
			for (k = 0; k < cbatch; k++)
				sum += inp[k * ocrow + i] * out[k * icrow + j];
			mweight[i * icrow + j] = momentum * mweight[i * icrow + j] + (1.0f - momentum) * (sum / cbatch);
			weight[i * icrow + j] -= lr * mweight[i * icrow + j];
		}
		sum = 0.0f;
		for (j = 0; j < cbatch; j++)
			sum += inp[j * ocrow + i];
		mbias[i] = momentum * mbias[i] + (1.0f - momentum) * (sum / cbatch);
		bias[i] -= lr * mbias[i];
	}

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < icrow; j++) {
			out[i * icrow + j] = 0.0f;
			for (k = 0; k < ocrow; k++)
				out[i * icrow + j] += copy_weight[k * icrow + j] * inp[i * ocrow + k];
		}
}

void Dense::test(const float* inp, float* out) {
	int i, j, k;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < ocrow; j++) {
			out[i * ocrow + j] = 0.0f;
			for (k = 0; k < icrow; k++)
				out[i * ocrow + j] += weight[j * icrow + k] * inp[i * icrow + k];
			out[i * ocrow + j] += bias[j];
		}
}

void Dense::save_to(std::ofstream &file) {
	file.write((char*)&Dense::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);

	file.write((char*)weight, sizeof(float) * (ocrow * icrow));
	file.write((char*)mweight, sizeof(float) * (ocrow * icrow));
	file.write((char*)copy_weight, sizeof(float) * (ocrow * icrow));

	file.write((char*)bias, sizeof(float) * ocrow);
	file.write((char*)mbias, sizeof(float) * ocrow);
}

void Dense::load_from(std::ifstream& file) {
	Layer::load_from(file);

	this->weight = init_mem(ocrow * icrow);
	this->mweight = init_mem(ocrow * icrow);
	this->copy_weight = init_mem(ocrow * icrow);
	this->bias = init_mem(ocrow);
	this->mbias = init_mem(ocrow);

	file.read((char*)weight, sizeof(float) * (ocrow * icrow));
	file.read((char*)mweight, sizeof(float) * (ocrow * icrow));
	file.read((char*)copy_weight, sizeof(float) * (ocrow * icrow));

	file.read((char*)bias, sizeof(float) * ocrow);
	file.read((char*)mbias, sizeof(float) * ocrow);
}
