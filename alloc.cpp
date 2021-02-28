#include "alloc.h"

float* init_mem(int len) {
	float* ptr = new float[len];

	for (int i = 0; i < len; i++)
		ptr[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;

	return ptr;
}

void copy_mem(const float* src, float* dst, int start, int len, int unit_size, int* shuffle_index) {
	if (shuffle_index)
		for (int i = 0; i < len; i++)
			for (int j = 0; j < unit_size; j++)
				dst[i * unit_size + j] = src[shuffle_index[start + i] * unit_size + j];
	else
		for (int i = 0; i < (len * unit_size); i++)
			dst[i] = src[start * unit_size + i];
}

void print_mem(const float* data, int crow, int ccol, int depth, int batch) {
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < depth; j++) {
			for (int k = 0; k < crow; k++) {
				for (int l = 0; l < ccol; l++)
					printf("%f,\t", data[i * (depth * ccol * crow) + j * (ccol * crow) + k * ccol + l]);
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
}
