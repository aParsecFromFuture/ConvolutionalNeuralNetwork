#include "cnn.h"

static int i, j, k, l, m, n, p, r;

float BATCH_DIV;
float LEARNING_RATE;
float MOMENTUM;

struct conv create_conv(int kcrow, int kccol, int kdepth, int icrow, int iccol, int idepth, int cbatch)
{
	struct conv layer;
	layer.kernel = init_mem(SHAPE4(kcrow, kccol, idepth, kdepth));
	layer.bias = init_mem(SHAPE1(kdepth));
	layer.copy_kernel = init_mem(SHAPE4(kcrow, kccol, idepth, kdepth));
	layer.mkernel = init_mem(SHAPE4(kcrow, kccol, idepth, kdepth));
	layer.mbias = init_mem(SHAPE1(kdepth));
	layer.kcrow = kcrow;
	layer.kccol = kccol;
	layer.kdepth = kdepth;
	layer.icrow = icrow;
	layer.iccol = iccol;
	layer.idepth = idepth;
	layer.ocrow = icrow - kcrow + 1;
	layer.occol = iccol - kccol + 1;
	layer.icr = icrow * iccol;
	layer.idcr = icrow * iccol * idepth;
	layer.kcr = kcrow * kccol;
	layer.kdcr = kcrow * kccol * idepth;
	layer.ocr = layer.ocrow * layer.occol;
	layer.odcr = layer.ocr * layer.kdepth;
	layer.cbatch = cbatch;

	for (i = 0; i < SHAPE4(kcrow, kccol, idepth, kdepth); i++)
		layer.mkernel[i] = 0.0f;

	for (i = 0; i < SHAPE1(kdepth); i++)
		layer.mbias[i] = 0.0f;

	return layer;
}

struct batchnorm create_batchnorm(int icrow, int iccol, int idepth, int cbatch)
{
	struct batchnorm layer;
	layer.icrow = icrow;
	layer.iccol = iccol;
	layer.idepth = idepth;
	layer.cbatch = cbatch;
	layer.icr = icrow * iccol;
	layer.idcr = icrow * iccol * idepth;
	layer.gamma = init_mem(SHAPE1(idepth));
	layer.mgamma = init_mem(SHAPE1(idepth));
	layer.beta = init_mem(SHAPE1(idepth));
	layer.mbeta = init_mem(SHAPE1(idepth));
	layer.mean = init_mem(SHAPE1(idepth));
	layer.var = init_mem(SHAPE1(idepth));
	layer.xhat = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.gammax = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.xmean = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.xmean2 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.std = init_mem(SHAPE1(idepth));
	layer.istd = init_mem(SHAPE1(idepth));

	layer.dgamma = init_mem(SHAPE1(idepth));
	layer.dbeta = init_mem(SHAPE1(idepth));
	layer.dxhat = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dmean = init_mem(SHAPE1(idepth));
	layer.dvar = init_mem(SHAPE1(idepth));
	layer.dstd = init_mem(SHAPE1(idepth));
	layer.distd = init_mem(SHAPE1(idepth));
	layer.dxmean = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dxmean2 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dxmean11 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dxmean12 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dx1 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	layer.dx2 = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));

	for (i = 0; i < idepth; i++) {
		layer.gamma[i] = 1.0f;
		layer.beta[i] = 0.0f;
		layer.mgamma[i] = 0.0f;
		layer.mbeta[i] = 0.0f;
	}

	return layer;
}

struct maxp create_maxp(int icrow, int iccol, int idepth, int cbatch)
{
	struct maxp layer;
	layer.icrow = icrow;
	layer.iccol = iccol;
	layer.idepth = idepth;
	layer.cbatch = cbatch;
	layer.ocrow = icrow / 2;
	layer.occol = iccol / 2;
	layer.icr = icrow * iccol;
	layer.idcr = layer.icr * layer.idepth;
	layer.ocr = layer.ocrow * layer.occol;
	layer.odcr = layer.ocr * layer.idepth;
	layer.maxmap = init_mem(SHAPE4(icrow, iccol, idepth, cbatch));
	return layer;
}

struct dense create_dense(int inp_nc, int out_nc, int cbatch)
{
	struct dense layer;
	layer.inp_nc = inp_nc;
	layer.out_nc = out_nc;
	layer.cbatch = cbatch;
	layer.weight = init_mem(SHAPE2(out_nc, inp_nc));
	layer.bias = init_mem(SHAPE1(out_nc));
	layer.copy_weight = init_mem(SHAPE2(out_nc, inp_nc));
	layer.mweight = init_mem(SHAPE2(out_nc, inp_nc));
	layer.mbias = init_mem(SHAPE1(out_nc));

	for (i = 0; i < (out_nc * inp_nc); i++)
		layer.mweight[i] = 0.0f;

	for (i = 0; i < out_nc; i++)
		layer.mbias[i] = 0.0f;

	return layer;
}

void f_relu(const float* inp, int len, float* out)
{
	for (i = 0; i < len; i++)
		out[i] = (inp[i] > 0.0f) ? inp[i] : 0.0f;
}

void f_tanh(const float* inp, int len, float* out)
{
	for (i = 0; i < len; i++)
		out[i] = 2.0f / (1.0f + pow(EULER, -2.0f * inp[i])) - 1.0f;
}

void f_conv(const float* inp, struct conv* layer, float* out)
{
	int icrow, iccol, idepth, cbatch, kcrow, kccol, ocrow, occol, odepth, idcr, icr, kdcr, kcr, odcr, ocr;
	const float* kernel, * bias;
	float sum;

	icrow = layer->icrow;
	iccol = layer->iccol;
	idepth = layer->idepth;
	cbatch = layer->cbatch;

	kcrow = layer->kcrow;
	kccol = layer->kccol;

	ocrow = layer->ocrow;
	occol = layer->occol;
	odepth = layer->kdepth;

	icr = layer->icr;
	idcr = layer->idcr;

	kcr = layer->kcr;
	kdcr = layer->kdcr;

	ocr = layer->ocr;
	odcr = layer->odcr;

	kernel = layer->kernel;
	bias = layer->bias;

	for(i = 0; i < cbatch; i++)
		for(j = 0; j < odepth; j++)
			for(k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++) {
					sum = 0.0f;
					for (m = 0; m < kcrow; m++)
						for (n = 0; n < kccol; n++)
							for (p = 0; p < idepth; p++)
								sum += inp[i * idcr + p * icr + (k + m) * iccol + (l + n)] * kernel[j * kdcr + p * kcr + m * kccol + n];
					out[i * odcr + j * ocr + k * occol + l] = sum + bias[j];
				}
}

void f_batchnorm(const float* inp, struct batchnorm* layer, float* out)
{
	int icrow, iccol, idepth, cbatch, icr, idcr;
	float* mean, * var, * gamma, * beta, * xhat;
	float* gammax, * istd, * std, * xmean, * xmean2;

	icrow = layer->icrow;
	iccol = layer->iccol;
	idepth = layer->idepth;
	cbatch = layer->cbatch;
	icr = layer->icr;
	idcr = layer->idcr;

	mean = layer->mean;
	var = layer->var;
	gamma = layer->gamma;
	beta = layer->beta;
	xhat = layer->xhat;

	gammax = layer->gammax;
	istd = layer->istd;
	std = layer->std;
	xmean = layer->xmean;
	xmean2 = layer->xmean2;

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

void f_maxp(const float* inp, struct maxp* layer, float* out)
{
	int icrow, iccol, depth, cbatch, ocrow, occol, idcr, icr, odcr, ocr, at, max_index;
	float* maxmap;
	float max;

	icrow = layer->icrow;
	iccol = layer->iccol;
	depth = layer->idepth;
	cbatch = layer->cbatch;
	ocrow = layer->ocrow;
	occol = layer->occol;

	icr = layer->icr;
	idcr = layer->idcr;
	ocr = layer->ocr;
	odcr = layer->odcr;

	maxmap = layer->maxmap;

	for(i = 0; i < cbatch; i++)
		for(j = 0; j < depth; j++)
			for(k = 0; k < ocrow; k++)
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

void f_dense(const float* inp, struct dense* layer, float* out)
{
	int wcrow, wccol, cbatch;
	const float * weight, * bias;

	wcrow = layer->out_nc;
	wccol = layer->inp_nc;
	cbatch = layer->cbatch;

	weight = layer->weight;
	bias = layer->bias;
	
	for(i = 0; i < cbatch; i++)
		for (j = 0; j < wcrow; j++) {
			out[i * wcrow + j] = 0.0f;
			for (k = 0; k < wccol; k++)
				out[i * wcrow + j] += weight[j * wccol + k] * inp[i * wccol + k];
			out[i * wcrow + j] += bias[j];
		}
}

void f_output(const float* inp, struct dense* layer, float* out)
{
	int wcrow, wccol, cbatch;
	const float* weight, * bias;
	float sum;

	wcrow = layer->out_nc;
	wccol = layer->inp_nc;
	cbatch = layer->cbatch;

	weight = layer->weight;
	bias = layer->bias;

	for(i = 0; i < cbatch; i++)
		for (j = 0; j < wcrow; j++) {
			out[i * wcrow + j] = 0.0f;
			for (k = 0; k < wccol; k++)
				out[i * wcrow + j] += weight[j * wccol + k] * inp[i * wccol + k];
			out[i * wcrow + j] += bias[j];
		}

	for (i = 0; i < cbatch; i++) {
		sum = 0.0f;

		for (j = 0; j < wcrow; j++)
			sum += pow(EULER, out[i * wcrow + j]);

		sum = 1.0f / sum;

		for (j = 0; j < wcrow; j++)
			out[i * wcrow + j] = pow(EULER, out[i * wcrow + j]) * sum;
	}
}

float evaluate(const float* output, const float* target, int category_count, int cbatch)
{
	float err = 0.0f;

	for (k = 0; k < cbatch; k++)
		for (l = 0; l < category_count; l++)
			err -= target[k * category_count + l] * log(output[k * category_count + l]);

	return err * BATCH_DIV;
}

void b_relu(const float* inp, int len, float* out)
{
	for (i = 0; i < len; i++)
		out[i] = (out[i] > 0.0f) ? inp[i] : 0.0f;
}

void b_tanh(const float* inp, int len, float* out)
{
	float tanh;
	
	for (i = 0; i < len; i++) {
		tanh = 2.0f / (1.0f + pow(EULER, -2.0f * out[i])) - 1.0f;
		out[i] = (1.0f - tanh * tanh) * inp[i];
	}
}
void b_conv(const float* inp, struct conv* layer, float* out)
{
	int icrow, iccol, idepth, cbatch, kcrow, kccol, ocrow, occol, odepth, idcr, icr, kdcr, kcr, odcr, ocr, shift_crow, shift_ccol, scrow, sccol;
	float* kernel, * bias, * copy_kernel, * mkernel, * mbias;
	float sum;

	icrow = layer->icrow;
	iccol = layer->iccol;
	idepth = layer->idepth;
	cbatch = layer->cbatch;

	kcrow = layer->kcrow;
	kccol = layer->kccol;

	ocrow = layer->ocrow;
	occol = layer->occol;
	odepth = layer->kdepth;

	icr = layer->icr;
	idcr = layer->idcr;

	kcr = layer->kcr;
	kdcr = layer->kdcr;

	ocr = layer->ocr;
	odcr = layer->odcr;

	kernel = layer->kernel;
	bias = layer->bias;
	copy_kernel = layer->copy_kernel;
	mkernel = layer->mkernel;
	mbias = layer->mbias;
	
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

					mkernel[i * kdcr + j * kcr + k * kccol + l] = MOMENTUM * mkernel[i * kdcr + j * kcr + k * kccol + l] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
					kernel[i * kdcr + j * kcr + k * kccol + l] -= LEARNING_RATE * mkernel[i * kdcr + j * kcr + k * kccol + l];
				}
		sum = 0.0f;
		for (j = 0; j < ocrow; j++)
			for (k = 0; k < occol; k++)
				for (l = 0; l < cbatch; l++)
					sum += inp[l * odcr + i * ocr + j * occol + k];
		mbias[i] = MOMENTUM * mbias[i] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
		bias[i] -= LEARNING_RATE * mbias[i];
	}
	
	shift_crow = icrow - ocrow;
	shift_ccol = iccol - occol;
	
	for(i = 0; i < cbatch; i++)
		for(j = 0; j < idepth; j++)
			for(k = 0; k < icrow; k++)
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

void b_batchnorm(const float* inp, struct batchnorm* layer, float* out)
{
	int icrow, iccol, idepth, cbatch, icr, idcr;
	float* mean, * var, * gamma, * beta, * xhat, * mgamma, * mbeta;
	float* gammax, * istd, * std, * xmean, * xmean2;
	float* dmean, * dvar, * dgamma, * dbeta, * dxhat, * dstd, * distd, * dxmean, * dxmean2, * dxmean11, * dxmean12, * dx1, * dx2;

	icrow = layer->icrow;
	iccol = layer->iccol;
	idepth = layer->idepth;
	cbatch = layer->cbatch;
	icr = layer->icr;
	idcr = layer->idcr;

	mean = layer->mean;
	var = layer->var;
	gamma = layer->gamma;
	beta = layer->beta;
	xhat = layer->xhat;
	mgamma = layer->mgamma;
	mbeta = layer->mbeta;

	gammax = layer->gammax;
	istd = layer->istd;
	std = layer->std;
	xmean = layer->xmean;
	xmean2 = layer->xmean2;

	dmean = layer->dmean;
	dvar = layer->dvar;
	dgamma = layer->dgamma;
	dbeta = layer->dbeta;
	dxhat = layer->dxhat;
	dstd = layer->dstd;
	distd = layer->distd;
	dxmean = layer->dxmean;
	dxmean2 = layer->dxmean2;
	dxmean11 = layer->dxmean11;
	dxmean12 = layer->dxmean12;
	dx1 = layer->dx1;
	dx2 = layer->dx2;

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

		mgamma[i] = MOMENTUM * mgamma[i] + (1.0f - MOMENTUM) * (dgamma[i] * BATCH_DIV / icr);
		gamma[i] -= LEARNING_RATE * mgamma[i];

		mbeta[i] = MOMENTUM * mbeta[i] + (1.0f - MOMENTUM) * (dbeta[i] * BATCH_DIV / icr);
		beta[i] -= LEARNING_RATE * mbeta[i];
	}
}

void b_maxp(const float* inp, struct maxp* layer, float* out)
{
	int icrow, iccol, depth, cbatch, ocrow, occol, icr, idcr, ocr, odcr;
	float* maxmap;

	icrow = layer->icrow;
	iccol = layer->iccol;
	depth = layer->idepth;
	cbatch = layer->cbatch;
	ocrow = layer->ocrow;
	occol = layer->occol;
	icr = layer->icr;
	idcr = layer->idcr;
	ocr = layer->ocr;
	odcr = layer->odcr;

	maxmap = layer->maxmap;

	for (i = 0; i < cbatch; i++)
		for (j = 0; j < depth; j++)
			for (k = 0; k < ocrow; k++)
				for (l = 0; l < occol; l++)
					for (m = 0; m < 2; m++)
						for (n = 0; n < 2; n++)
							if (maxmap[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] > 0.5f)
								out[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] = inp[i * odcr + j * ocr + k * occol + l];
							else
								out[i * idcr + j * icr + (2 * k + m) * iccol + (2 * l + n)] = 0.0f;
}

void b_dense(const float* inp, struct dense* layer, float* out)
{
	int wcrow, wccol, cbatch;
	float* weight, * bias, * copy_weight, * mweight, * mbias;
	float sum;

	wcrow = layer->out_nc;
	wccol = layer->inp_nc;
	cbatch = layer->cbatch;

	weight = layer->weight;
	copy_weight = layer->copy_weight;
	bias = layer->bias;
	mweight = layer->mweight;
	mbias = layer->mbias;
	
	for (i = 0; i < (wcrow * wccol); i++)
		copy_weight[i] = weight[i];
	
	for (i = 0; i < wcrow; i++) {
		for (j = 0; j < wccol; j++) {
			sum = 0.0f;
			for (k = 0; k < cbatch; k++)
				sum += inp[k * wcrow + i] * out[k * wccol + j];
			mweight[i * wccol + j] = MOMENTUM * mweight[i * wccol + j] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
			weight[i * wccol + j] -= LEARNING_RATE * mweight[i * wccol + j];
		}
		sum = 0.0f;
		for (j = 0; j < cbatch; j++)
			sum += inp[j * wcrow + i];
		mbias[i] = MOMENTUM * mbias[i] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
		bias[i] -= LEARNING_RATE * mbias[i];
	}
	
	for(i = 0; i < cbatch; i++)
		for (j = 0; j < wccol; j++) {
			out[i * wccol + j] = 0.0f;
			for (k = 0; k < wcrow; k++)
				out[i * wccol + j] += copy_weight[k * wccol + j] * inp[i * wcrow + k];
		}
}

void b_output(const float* inp, const float* target, struct dense* layer, float* out)
{
	int wcrow, wccol, cbatch;
	float* weight, * bias, * copy_weight, * mweight, * mbias;
	float sum;

	wcrow = layer->out_nc;
	wccol = layer->inp_nc;
	cbatch = layer->cbatch;

	weight = layer->weight;
	copy_weight = layer->copy_weight;
	bias = layer->bias;
	mweight = layer->mweight;
	mbias = layer->mbias;

	for (i = 0; i < (wcrow * wccol); i++)
		copy_weight[i] = weight[i];
	
	for (i = 0; i < wcrow; i++) {
		for (j = 0; j < wccol; j++) {
			sum = 0.0f;
			for (k = 0; k < cbatch; k++)
				sum += (inp[k * wcrow + i] - target[k * wcrow + i]) * out[k * wccol + j];
			mweight[i * wccol + j] = MOMENTUM * mweight[i * wccol + j] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
			weight[i * wccol + j] -= LEARNING_RATE * mweight[i * wccol + j];
		}
		sum = 0.0f;
		for (j = 0; j < cbatch; j++)
			sum += inp[i * cbatch + j];
		mbias[i] = MOMENTUM * mbias[i] + (1.0f - MOMENTUM) * (sum * BATCH_DIV);
		bias[i] -= LEARNING_RATE * mbias[i];
	}
	
	for (i = 0; i < cbatch; i++)
		for (j = 0; j < wccol; j++) {
			out[i * wccol + j] = 0.0f;
			for (k = 0; k < wcrow; k++)
				out[i * wccol + j] += copy_weight[k * wccol + j] * (inp[i * wcrow + k] - target[i * wcrow + k]);
		}
}
