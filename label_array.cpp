#include "label_array.h"

LabelArray::LabelArray(int category_count) {
	this->category_count = category_count;
	this->label_count = 0;
	this->data = 0;
}

LabelArray::~LabelArray() {
	if (data)
		delete[] data;
}

void LabelArray::load_from(const char* file_path) {
	std::ifstream file(file_path, std::ios::binary);
	int tmp;
	int* tmp_data;

	if (!file.is_open()) {
		printf("The file \"%s\" couldn't open\n", file_path);
		exit(1);
	}

	label_count = 0;
	while (file.read(reinterpret_cast<char*>(&tmp), sizeof(int)))
		label_count++;

	file.clear();
	file.seekg(0);

	if (data)
		delete[] data;

	tmp_data = new int[label_count];
	data = new float[label_count * category_count];

	file.read(reinterpret_cast<char*>(tmp_data), sizeof(int) * label_count);
	file.close();

	int k = 0;
	for (int i = 0; i < label_count; i++)
		for (int j = 0; j < category_count; j++)
			data[k++] = (j == tmp_data[i] - 1) ? 1.0f : 0.0f;

	delete[] tmp_data;
}

void LabelArray::alloc(int label_count) {
	if (data)
		delete[] data;

	this->label_count = label_count;

	data = new float[label_count * category_count];
	for (int i = 0; i < (label_count * category_count); i++)
		data[i] = 0.0f;
}

int* LabelArray::simplify() {
	int* prediction = new int[label_count];
	int max_index;
	float max;

	for (int i = 0; i < label_count; i++) {
		max_index = 0;
		max = 0.0f;
		for(int j = 0; j < category_count; j++)
			if (max < data[i * category_count + j]) {
				max = data[i * category_count + j];
				max_index = j;
			}
		prediction[i] = max_index;
	}

	return prediction;
}

float* LabelArray::get_data(int order) const {
	return &data[order * category_count];
}

float* LabelArray::raw() {
	return data;
}

int LabelArray::item_size() const {
	return category_count;
}
int LabelArray::length() const {
	return label_count;
}