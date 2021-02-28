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
	int tmp, count = 0;
	int* tmp_data;

	while (file.read(reinterpret_cast<char*>(&tmp), sizeof(int)))
		count++;

	file.clear();
	file.seekg(0);

	if (data)
		delete[] data;

	tmp_data = new int[count];
	data = new float[count * category_count];

	file.read(reinterpret_cast<char*>(tmp_data), sizeof(int) * count);
	file.close();

	label_count = count / category_count;

	int k = 0;
	for (int i = 0; i < count; i++)
		for (int j = 0; j < category_count; j++)
			data[k++] = (j == tmp_data[i] - 1) ? 1.0f : 0.0f;

	delete[] tmp_data;
}

float* LabelArray::get_data(int order) const {
	return &data[order * category_count];
}
