#include "preprocessing.h"

int* create_shuffle_index(int len)
{
	int* shuffle_index = (int*)malloc(sizeof(int) * len);

	for (int i = 0; i < len; i++)
		shuffle_index[i] = i;

	return shuffle_index;
}

void shuffle(int* shuffle_index, int len)
{
	int to, swap;

	for (int i = 0; i < len; i++) {
		to = rand() % len;
		swap = shuffle_index[i];
		shuffle_index[i] = shuffle_index[to];
		shuffle_index[to] = swap;
	}
}

