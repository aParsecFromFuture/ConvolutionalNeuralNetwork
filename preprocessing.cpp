#include "preprocessing.h"

const float EULER = 2.7182818f;
const float RAND_DIV = 1.0f / RAND_MAX;
const float EPSILON = 1e-5;

static int i, j, k, l;

float* init_mem(int len)
{
	float* ptr = (float*)malloc(sizeof(float) * len);

	for (i = 0; i < len; i++)
		ptr[i] = (float)rand() * RAND_DIV - 0.5f;

	return ptr;
}

void copy_mem(const float* src, float* dst, int start, int len, int unit_size, int* shuffle_index)
{
	if (shuffle_index)
		for (i = 0; i < len; i++)
			for (j = 0; j < unit_size; j++)
				dst[i * unit_size + j] = src[shuffle_index[start + i] * unit_size + j];
	else
		for (i = 0; i < (len * unit_size); i++)
			dst[i] = src[start * unit_size + i];
}

float* create_target(int crow, int ccol, int* data)
{
	float* target = new float[crow * ccol];
	k = 0;
	for (i = 0; i < ccol; i++)
		for (j = 0; j < crow; j++)
			target[k++] = (j == data[i] - 1) ? 1.0f : 0.0f;
	return target;
}

void print_mem(const float* data, int crow, int ccol, int depth, int batch)
{
	for (i = 0; i < batch; i++) {
		for (j = 0; j < depth; j++) {
			for (k = 0; k < crow; k++) {
				for (l = 0; l < ccol; l++)
					printf("%f,\t", data[i * (depth * ccol * crow) + j * (ccol * crow) + k * ccol + l]);
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
}

float* read_data(const char* file_path)
{
	
	FILE* fp;
	int i, count;
	float tmp;
	float* dataset;

	fopen_s(&fp, file_path, "r");

	count = 0;

	while (fscanf_s(fp, "%f", &tmp) == 1) count++;

	dataset = (float*)malloc(sizeof(float) * count);

	fseek(fp, 0, SEEK_SET);

	for (i = 0; i < count; i++)
		fscanf_s(fp, "%f", &dataset[i]);

	fclose(fp);

	return dataset;
}

int* read_idata(const char* file_path)
{
	FILE* fp;
	int i, tmp, count;
	int* data;

	fopen_s(&fp, file_path, "r");

	count = 0;

	while (fscanf_s(fp, "%d", &tmp) == 1) count++;

	data = (int*)malloc(sizeof(int) * count);

	fseek(fp, 0, SEEK_SET);

	for (i = 0; i < count; i++)
		fscanf_s(fp, "%d", &data[i]);

	fclose(fp);

	return data;
}

float* batch_normalization_train(float* data, int sample_count, int feature_dim, float* mean, float* variance) {
	float* normalized_data = (float*)malloc(sizeof(float) * sample_count * feature_dim);
	int i, j;

	for (i = 0; i < feature_dim; i++)
		mean[i] = variance[i] = 0.0f;

	for (i = 0; i < sample_count; i++)
		for (j = 0; j < feature_dim; j++)
			mean[j] += data[i * feature_dim + j];

	for (i = 0; i < feature_dim; i++)
		mean[i] /= sample_count;

	for (i = 0; i < sample_count; i++)
		for (j = 0; j < feature_dim; j++)
			variance[j] += (data[i * feature_dim + j] - mean[j]) * (data[i * feature_dim + j] - mean[j]);

	for (i = 0; i < feature_dim; i++)
		variance[i] /= sample_count;

	for (i = 0; i < sample_count; i++)
		for (j = 0; j < feature_dim; j++)
			normalized_data[i * feature_dim + j] = (data[i * feature_dim + j] - mean[j]) / (sqrt(variance[j] + EPSILON));

	return normalized_data;
}

float* batch_normalization_test(float* data, int sample_count, int feature_dim, float* mean, float* variance) {
	float* normalized_data = (float*)malloc(sizeof(float) * sample_count * feature_dim);
	int i, j;

	for (i = 0; i < sample_count; i++)
		for (j = 0; j < feature_dim; j++)
			normalized_data[i * feature_dim + j] = (data[i * feature_dim + j] - mean[j]) / (sqrt(variance[j] + EPSILON));

	return normalized_data;
}

int* create_shuffle_index(int len)
{
	int* shuffle_index = (int*)malloc(sizeof(int) * len);

	for (i = 0; i < len; i++)
		shuffle_index[i] = i;

	return shuffle_index;
}

void shuffle(int* shuffle_index, int len)
{
	int to, swap;

	for (i = 0; i < len; i++) {
		to = rand() % len;
		swap = shuffle_index[i];
		shuffle_index[i] = shuffle_index[to];
		shuffle_index[to] = swap;
	}
}

