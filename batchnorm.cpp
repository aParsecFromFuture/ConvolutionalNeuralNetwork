#include "batchnorm.h"

BatchNorm::BatchNorm() {
	this->gamma = 0;
	this->mgamma = 0;
	this->beta = 0;
	this->mbeta = 0;
	this->mean = 0;
	this->var = 0;
	this->xhat = 0;
	this->gammax = 0;
	this->xmean = 0;
	this->xmean2 = 0;
	this->std = 0;
	this->istd = 0;

	this->dgamma = 0;
	this->dbeta = 0;
	this->dxhat = 0;
	this->dmean = 0;
	this->dvar = 0;
	this->dstd = 0;
	this->distd = 0;
	this->dxmean = 0;
	this->dxmean2 = 0;
	this->dxmean11 = 0;
	this->dxmean12 = 0;
	this->dx1 = 0;
	this->dx2 = 0;
}

void BatchNorm::init(int icrow, int iccol, int idepth, int batch_size) {
	Layer::init(icrow, iccol, idepth, icrow, iccol, idepth, batch_size);

	this->gamma = init_mem(this->idepth);
	this->mgamma = init_mem(this->idepth);
	this->beta = init_mem(this->idepth);
	this->mbeta = init_mem(this->idepth);
	this->mean = init_mem(this->idepth);
	this->var = init_mem(this->idepth);
	this->xhat = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->gammax = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->xmean = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->xmean2 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->std = init_mem(this->idepth);
	this->istd = init_mem(this->idepth);

	this->dgamma = init_mem(this->idepth);
	this->dbeta = init_mem(this->idepth);
	this->dxhat = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dmean = init_mem(this->idepth);
	this->dvar = init_mem(this->idepth);
	this->dstd = init_mem(this->idepth);
	this->distd = init_mem(this->idepth);
	this->dxmean = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dxmean2 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dxmean11 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dxmean12 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dx1 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);
	this->dx2 = init_mem(this->icrow * this->iccol * this->idepth * this->cbatch);

	for (int i = 0; i < this->idepth; i++) {
		this->gamma[i] = 1.0f;
		this->beta[i] = 0.0f;
		this->mgamma[i] = 0.0f;
		this->mbeta[i] = 0.0f;
	}
}

void BatchNorm::feedforward(const float* inp, float* out) {
	int i, j, k;

	for (i = 0; i < idepth; i++) {
		mean[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				mean[i] += inp[j * idcr + i * icr + k];
		mean[i] = mean[i] / (icr * cbatch);

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++) {
				xmean[j * idcr + i * icr + k] = inp[j * idcr + i * icr + k] - mean[i];
				xmean2[j * idcr + i * icr + k] = xmean[j * idcr + i * icr + k] * xmean[j * idcr + i * icr + k];
			}

		var[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				var[i] += xmean2[j * idcr + i * icr + k];
		var[i] = var[i] / (icr * cbatch);

		std[i] = sqrt(var[i] + EPSILON);
		istd[i] = 1.0f / std[i];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				xhat[j * idcr + i * icr + k] = xmean[j * idcr + i * icr + k] * istd[i];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				gammax[j * idcr + i * icr + k] = gamma[i] * xhat[j * idcr + i * icr + k];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				out[j * idcr + i * icr + k] = gammax[j * idcr + i * icr + k] + beta[i];
	}
}

void BatchNorm::backpropagation(const float* inp, const float* target, float* out) {
	int i, j, k;
	
	for (i = 0; i < idepth; i++) {
		dbeta[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dbeta[i] += inp[j * idcr + i * icr + k];

		dgamma[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dgamma[i] += inp[j * idcr + i * icr + k] * xhat[j * idcr + i * icr + k];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dxhat[j * idcr + i * icr + k] = inp[j * idcr + i * icr + k] * gamma[i];

		distd[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				distd[i] += dxhat[j * idcr + i * icr + k] * xmean[j * idcr + i * icr + k];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dxmean11[j * idcr + i * icr + k] = dxhat[j * idcr + i * icr + k] * istd[i];

		dstd[i] = -1.0f / (std[i] * std[i]) * distd[i];
		dvar[i] = 0.5f / std[i] * dstd[i];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dxmean2[j * idcr + i * icr + k] = dvar[i] / (icr * cbatch);

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dxmean12[j * idcr + i * icr + k] = 2.0f * xmean[j * idcr + i * icr + k] * dxmean2[j * idcr + i * icr + k];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dx1[j * idcr + i * icr + k] = dxmean11[j * idcr + i * icr + k] + dxmean12[j * idcr + i * icr + k];

		dmean[i] = 0.0f;
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dmean[i] += dxmean11[j * idcr + i * icr + k] + dxmean12[j * idcr + i * icr + k];
		dmean[i] = -dmean[i];

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				dx2[j * idcr + i * icr + k] = dmean[i] / (icr * cbatch);

		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				out[j * idcr + i * icr + k] = dx1[j * idcr + i * icr + k] + dx2[j * idcr + i * icr + k];
		
		mgamma[i] = momentum * mgamma[i] + (1.0f - momentum) * (dgamma[i] / (cbatch * icr));
		gamma[i] -= lr * mgamma[i];

		mbeta[i] = momentum * mbeta[i] + (1.0f - momentum) * (dbeta[i] / (cbatch * icr));
		beta[i] -= lr * mbeta[i];
	}
}

void BatchNorm::test(const float* inp, float* out) {
	int i, j, k;
	
	for (i = 0; i < idepth; i++)
		for (j = 0; j < cbatch; j++)
			for (k = 0; k < icr; k++)
				out[j * idcr + i * icr + k] = gamma[i] * ((inp[j * idcr + i * icr + k] - mean[i]) * istd[i]) + beta[i];
}

void BatchNorm::save_to(std::ofstream &file) {
	file.write((char*)&BatchNorm::SERIALIZE_ID, sizeof(int));

	Layer::save_to(file);

	file.write((char*)gamma, sizeof(float) * idepth);
	file.write((char*)mgamma, sizeof(float) * idepth);
	file.write((char*)beta, sizeof(float) * idepth);
	file.write((char*)mbeta, sizeof(float) * idepth);
	file.write((char*)mean, sizeof(float) * idepth);
	file.write((char*)var, sizeof(float) * idepth);
	file.write((char*)std, sizeof(float) * idepth);
	file.write((char*)istd, sizeof(float) * idepth);
	file.write((char*)dgamma, sizeof(float) * idepth);
	file.write((char*)dbeta, sizeof(float) * idepth);
	file.write((char*)dmean, sizeof(float) * idepth);
	file.write((char*)dvar, sizeof(float) * idepth);
	file.write((char*)dstd, sizeof(float) * idepth);
	file.write((char*)distd, sizeof(float) * idepth);

	file.write((char*)xhat, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)gammax, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)xmean, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)xmean2, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dxhat, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dxmean, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dxmean2, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dxmean11, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dxmean12, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dx1, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.write((char*)dx2, sizeof(float) * (icrow * iccol * idepth * cbatch));
}

void BatchNorm::load_from(std::ifstream &file) {
	Layer::load_from(file);

	this->gamma = init_mem(idepth);
	this->mgamma = init_mem(idepth);
	this->beta = init_mem(idepth);
	this->mbeta = init_mem(idepth);
	this->mean = init_mem(idepth);
	this->var = init_mem(idepth);
	this->xhat = init_mem(icrow * iccol * idepth * cbatch);
	this->gammax = init_mem(icrow * iccol * idepth * cbatch);
	this->xmean = init_mem(icrow * iccol * idepth * cbatch);
	this->xmean2 = init_mem(icrow * iccol * idepth * cbatch);
	this->std = init_mem(idepth);
	this->istd = init_mem(idepth);

	this->dgamma = init_mem(idepth);
	this->dbeta = init_mem(idepth);
	this->dxhat = init_mem(icrow * iccol * idepth * cbatch);
	this->dmean = init_mem(idepth);
	this->dvar = init_mem(idepth);
	this->dstd = init_mem(idepth);
	this->distd = init_mem(idepth);
	this->dxmean = init_mem(icrow * iccol * idepth * cbatch);
	this->dxmean2 = init_mem(icrow * iccol * idepth * cbatch);
	this->dxmean11 = init_mem(icrow * iccol * idepth * cbatch);
	this->dxmean12 = init_mem(icrow * iccol * idepth * cbatch);
	this->dx1 = init_mem(icrow * iccol * idepth * cbatch);
	this->dx2 = init_mem(icrow * iccol * idepth * cbatch);

	file.read((char*)gamma, sizeof(float) * idepth);
	file.read((char*)mgamma, sizeof(float) * idepth);
	file.read((char*)beta, sizeof(float) * idepth);
	file.read((char*)mbeta, sizeof(float) * idepth);
	file.read((char*)mean, sizeof(float) * idepth);
	file.read((char*)var, sizeof(float) * idepth);
	file.read((char*)std, sizeof(float) * idepth);
	file.read((char*)istd, sizeof(float) * idepth);
	file.read((char*)dgamma, sizeof(float) * idepth);
	file.read((char*)dbeta, sizeof(float) * idepth);
	file.read((char*)dmean, sizeof(float) * idepth);
	file.read((char*)dvar, sizeof(float) * idepth);
	file.read((char*)dstd, sizeof(float) * idepth);
	file.read((char*)distd, sizeof(float) * idepth);

	file.read((char*)xhat, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)gammax, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)xmean, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)xmean2, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dxhat, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dxmean, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dxmean2, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dxmean11, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dxmean12, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dx1, sizeof(float) * (icrow * iccol * idepth * cbatch));
	file.read((char*)dx2, sizeof(float) * (icrow * iccol * idepth * cbatch));
}