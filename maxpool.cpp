#include "maxpool.h"

MaxPool::MaxPool() {
	this->maxmap = 0;
}

void MaxPool::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow / 2, iccol / 2, idepth, batch_size);
	this->maxmap = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
}

void MaxPool::feedforward(const float* inp, float* out) {
	int i, j, k, l;
	int at, max_index;
	float max;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < odepth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++) {
					max_index = i * idcr + j * icr + (2 * k) * iccol + (2 * l);
					max = inp[max_index];
					maxmap[max_index] = 0.0f;

					at = i * idcr + j * icr + (2 * k) * iccol + (2 * l + 1);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}
					maxmap[at] = 0.0f;

					at = i * idcr + j * icr + (2 * k + 1) * iccol + (2 * l);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}
					maxmap[at] = 0.0f;

					at = i * idcr + j * icr + (2 * k + 1) * iccol + (2 * l + 1);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}
					maxmap[at] = 0.0f;

					out[i * odcr + j * ocr + k * occol + l] = max;
					maxmap[max_index] = 1.0f;
				}
}

void MaxPool::backpropagation(const float* inp, const float* target, float* out) {
	int i, j, k, l, m, n;
	
	for (i = 0; i < cbatch; i++)
		for (j = 0; j < odepth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++)
					for (m = 0; m < 2; m++)
						for (n = 0; n < 2; n++)
							if (maxmap[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] > 0.5f)
								out[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] = inp[i * odcr + j * ocr + k * occol + l];
							else
								out[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] = 0.0f;
}

void MaxPool::test(const float* inp, float* out) {
	int i, j, k, l;
	int at, max_index;
	float max;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < odepth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++) {
					max_index = i * idcr + j * icr + (2 * k) * iccol + (2 * l);
					max = inp[max_index];

					at = i * idcr + j * icr + (2 * k) * iccol + (2 * l + 1);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}

					at = i * idcr + j * icr + (2 * k + 1) * iccol + (2 * l);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}

					at = i * idcr + j * icr + (2 * k + 1) * iccol + (2 * l + 1);
					if (max < inp[at]) {
						max_index = at;
						max = inp[at];
					}

					out[i * odcr + j * ocr + k * occol + l] = max;
				}
}

void MaxPool::save_to(std::ofstream &file) {
	file.write((char*)&MaxPool::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);

	file.write((char*)maxmap, sizeof(float) * (icrow * iccol * idepth * cbatch));
}

void MaxPool::load_from(std::ifstream& file) {
	Layer::load_from(file);

	this->maxmap = init_mem(icrow * iccol * idepth * cbatch);

	file.read((char*)maxmap, sizeof(float) * (icrow * iccol * idepth * cbatch));
}
