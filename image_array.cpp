#include "image_array.h"

ImageArray::ImageArray(int width, int height, int channel) {
	this->width = width;
	this->height = height;
	this->channel = channel;
	this->image_count = 0;
	this->data = 0;
}

ImageArray::~ImageArray() {

}

void ImageArray::load_from(const char* file_path) {
	std::ifstream file(file_path, std::ios::binary);
	int count = 0;
	float tmp;
	
	while (file.read(reinterpret_cast<char*>(&tmp), sizeof(float)))
		count++;

	file.clear();
	file.seekg(0);
	
	if (data)
		delete[] data;

	data = new float[count];

	while (file.read(reinterpret_cast<char*>(data), sizeof(float) * count));
	file.close();

	image_count = count / (width * height * channel);
}

void ImageArray::alloc(int image_count) {
	if (data)
		delete[] data;

	this->image_count = image_count;

	data = new float[image_count * width * height * channel];
	for (int i = 0; i < (image_count * width * height * channel); i++)
		data[i] = 0.0f;
}

int ImageArray::item_size() const {
	return width * height * channel;
}

int ImageArray::count() const {
	return image_count;
}

float* ImageArray::get_data(int order) const {
	return &data[order * (width * height * channel)];
}

float* ImageArray::raw() {
	return data;
}

void ImageArray::batch_normalization(ImageArray& train_images, ImageArray& test_images) {
	int feature_dim = train_images.width * train_images.height * train_images.channel;
	float* mean = new float[feature_dim];
	float* variance = new float[feature_dim];

	for (int i = 0; i < feature_dim; i++)
		mean[i] = variance[i] = 0.0f;

	for (int i = 0; i < train_images.image_count; i++)
		for (int j = 0; j < feature_dim; j++)
			mean[j] += train_images.data[i * feature_dim + j];

	for (int i = 0; i < feature_dim; i++)
		mean[i] /= train_images.image_count;

	for (int i = 0; i < train_images.image_count; i++)
		for (int j = 0; j < feature_dim; j++)
			variance[j] += (train_images.data[i * feature_dim + j] - mean[j]) * (train_images.data[i * feature_dim + j] - mean[j]);

	for (int i = 0; i < feature_dim; i++)
		variance[i] /= train_images.image_count;

	for (int i = 0; i < train_images.image_count; i++)
		for (int j = 0; j < feature_dim; j++)
			train_images.data[i * feature_dim + j] = (train_images.data[i * feature_dim + j] - mean[j]) / (sqrt(variance[j] + 0.0001f));

	for (int i = 0; i < test_images.image_count; i++)
		for (int j = 0; j < feature_dim; j++)
			test_images.data[i * feature_dim + j] = (test_images.data[i * feature_dim + j] - mean[j]) / (sqrt(variance[j] + 0.0001f));
}

void ImageArray::min_max_scaling(ImageArray& images) {
	int len = (images.width * images.height * images.channel * images.image_count);

	for (int i = 0; i < len; i++)
		images.data[i] = images.data[i] / 255.0f;
}