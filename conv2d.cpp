#include "conv2d.h"

Conv2D::Conv2D() {
	this->kcrow = 0;
	this->kccol = 0;
	this->kdepth = 0;

	this->kcr = 0;
	this->kdcr = 0;

	this->kernel = 0;
	this->mkernel = 0;
	this->copy_kernel = 0;
	this->bias = 0;
	this->mbias = 0;
}

Conv2D::Conv2D(int kcrow, int kccol, int kdepth) {
	this->kcrow = kcrow;
	this->kccol = kccol;
	this->kdepth = kdepth;
	
	this->kcr = kcrow * kccol;

	this->kernel = 0;
	this->mkernel = 0;
	this->copy_kernel = 0;
	this->bias = 0;
	this->mbias = 0;
}

void Conv2D::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow - this->kcrow + 1, iccol - this->kccol + 1, this->kdepth, batch_size);

	this->kdcr = this->kcr * idepth;

	this->kernel = init_mem(this->kcrow * this->kccol * this->idepth * this->kdepth);
	this->bias = init_mem(this->kdepth);
	this->copy_kernel = init_mem(this->kcrow * this->kccol * this->idepth * this->kdepth);
	this->mkernel = init_mem(this->kcrow * this->kccol * this->idepth * this->kdepth);
	this->mbias = init_mem(this->kdepth);

	for (int i = 0; i < (this->kcrow * this->kccol * this->idepth * this->kdepth); i++)
		this->mkernel[i] = 0.0f;

	for (int i = 0; i < kdepth; i++)
		this->mbias[i] = 0.0f;
}

void Conv2D::feedforward(const float* inp, float* out) {
	int i, j, k, l, m, n, p;
	float sum;
	
	for (i = 0; i < cbatch; i++)
		for (j = 0; j < odepth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++) {
					sum = 0.0f;
					for (m = 0; m < kcrow; m++)
						for (n = 0; n < kccol; n++)
							for (p = 0; p < idepth; p++)
								sum += inp[i * idcr + p * icr + (k + m) * iccol + (l + n)] * kernel[j * kdcr + p * kcr + m * kccol + n];
					out[i * odcr + j * ocr + k * occol + l] = sum + bias[j];
				}
}

void Conv2D::backpropagation(const float* inp, const float* target, float* out) {
	int i, j, k, l, m, n, p;
	int shift_crow, shift_ccol, scrow, sccol;
	float sum;

	for (i = 0; i < (odepth * kdcr); i++)
		copy_kernel[i] = kernel[i];

	for (i = 0; i < odepth; i++) {
		for (j = 0; j < idepth; j++)
			for (k = 0; k < kcrow; k++)
				for (l = 0; l < kccol; l++) {
					sum = 0.0f;
					for (m = 0; m < cbatch; m++)
						for (n = 0; n < ocrow; n++)
							for (p = 0; p < occol; p++)
								sum += out[m * idcr + j * icr + (k + n) * iccol + (l + p)] * inp[m * odcr + i * ocr + n * occol + p];

					mkernel[i * kdcr + j * kcr + k * kccol + l] = momentum * mkernel[i * kdcr + j * kcr + k * kccol + l] + (1.0f - momentum) * (sum / cbatch);
					kernel[i * kdcr + j * kcr + k * kccol + l] -= lr * mkernel[i * kdcr + j * kcr + k * kccol + l];
				}
		sum = 0.0f;
		for (j = 0; j < ocrow; j++)
			for (k = 0; k < occol; k++)
				for (l = 0; l < cbatch; l++)
					sum += inp[l * odcr + i * ocr + j * occol + k];
		mbias[i] = momentum * mbias[i] + (1.0f - momentum) * (sum / cbatch);
		bias[i] -= lr * mbias[i];
	}

	shift_crow = icrow - ocrow;
	shift_ccol = iccol - occol;
	
	for (i = 0; i < cbatch; i++)
		for (j = 0; j < idepth; j++)
			for (k = 0; k < icrow; k++)
				for (l = 0; l < iccol; l++) {
					sum = 0.0f;
					for (m = 0; m < odepth; m++)
						for (n = 0; n < kcrow; n++)
							for (p = 0; p < kccol; p++) {
								scrow = k + n - shift_crow;
								sccol = l + p - shift_ccol;
								if (scrow >= 0 && sccol >= 0 && scrow < ocrow && sccol < occol)
									sum += copy_kernel[m * kdcr + j * kcr + (kcrow - n - 1) * kccol + (kccol - p - 1)] * inp[i * odcr + m * ocr + scrow * occol + sccol];
							}
					out[i * idcr + j * icr + k * iccol + l] = sum;
				}
}

void Conv2D::test(const float* inp, float* out, int csample) {
	int i, j, k, l, m, n, p;
	float sum;

	for (i = 0; i < csample; i++)
		for (j = 0; j < odepth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++) {
					sum = 0.0f;
					for (m = 0; m < kcrow; m++)
						for (n = 0; n < kccol; n++)
							for (p = 0; p < idepth; p++)
								sum += inp[i * idcr + p * icr + (k + m) * iccol + (l + n)] * kernel[j * kdcr + p * kcr + m * kccol + n];
					out[i * odcr + j * ocr + k * occol + l] = sum + bias[j];
				}
}

void Conv2D::save_to(std::ofstream &file) {
	file.write((char*)&Conv2D::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);

	file.write((char*)&kcrow, sizeof(kcrow));
	file.write((char*)&kccol, sizeof(kccol));
	file.write((char*)&kdepth, sizeof(kdepth));
	file.write((char*)&kcr, sizeof(kcr));
	file.write((char*)&kdcr, sizeof(kdcr));

	file.write((char*)kernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));
	file.write((char*)mkernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));
	file.write((char*)copy_kernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));

	file.write((char*)bias, sizeof(float) * kdepth);
	file.write((char*)mbias, sizeof(float) * kdepth);
}

void Conv2D::load_from(std::ifstream& file) {
	Layer::load_from(file);
	
	file.read((char*)&kcrow, sizeof(kcrow));
	file.read((char*)&kccol, sizeof(kccol));
	file.read((char*)&kdepth, sizeof(kdepth));
	file.read((char*)&kcr, sizeof(kcr));
	file.read((char*)&kdcr, sizeof(kdcr));

	this->kernel = init_mem(kcrow * kccol * idepth * kdepth);
	this->bias = init_mem(kdepth);
	this->copy_kernel = init_mem(kcrow * kccol * idepth * kdepth);
	this->mkernel = init_mem(kcrow * kccol * idepth * kdepth);
	this->mbias = init_mem(kdepth);
	
	file.read((char*)kernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));
	file.read((char*)mkernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));
	file.read((char*)copy_kernel, sizeof(float) * (kcrow * kccol * idepth * kdepth));

	file.read((char*)bias, sizeof(float) * kdepth);
	file.read((char*)mbias, sizeof(float) * kdepth);
}
