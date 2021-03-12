#include "ann.h"

ANN::ANN() {
	layer_count = 0;
	batch_size = 0;
	category_count = 0;
	layer = 0;
}

ANN::~ANN() {

}

void ANN::add_layer(Layer* ly) {
	Layer** tmp_layer = new Layer * [layer_count + 1];

	for (int i = 0; i < layer_count; i++)
		tmp_layer[i] = layer[i];
	delete[] layer;

	layer = tmp_layer;
	layer[layer_count++] = ly;
}

void ANN::setup(int width, int height, int channel, int batch_size, int category_count) {
	this->batch_size = batch_size;
	this->category_count = category_count;
	
	layer[0]->init(width, height, channel, batch_size);

	for (int i = 1; i < layer_count; i++)
		layer[i]->init(layer[i - 1]->ocrow, layer[i - 1]->occol, layer[i - 1]->odepth, batch_size);
}

void ANN::train(const ImageArray &images, const LabelArray &labels, int epoch, float lr, float valid_split, float momentum) {
	int train_image_count, valid_image_count, train_batch_count, valid_batch_count, image_size;
	int* shuffle_array;
	float err, err_now, * target, ** output;

	output = new float* [layer_count + 1];
	output[0] = init_mem(layer[0]->icrow * layer[0]->iccol * layer[0]->idepth * batch_size);
	for (int i = 0; i < layer_count; i++)
		output[i + 1] = init_mem(layer[i]->ocrow * layer[i]->occol * layer[i]->odepth * batch_size);

	train_image_count = (images.image_count * (1.0f - valid_split));
	train_batch_count = train_image_count / batch_size;
	valid_image_count = (images.image_count * valid_split);
	valid_batch_count = valid_image_count / batch_size;
	shuffle_array = create_shuffle_index(train_image_count);
	image_size = images.item_size();

	target = init_mem(category_count * batch_size);

	for (int i = 0; i < layer_count; i++) {
		layer[i]->lr = lr;
		layer[i]->momentum = momentum;
	}

	for (int i = 0; i < epoch; i++) {
		shuffle(shuffle_array, train_image_count);
		err = 0.0f;
		for (int j = 0; j < train_batch_count; j++) {
			copy_mem(images.get_data(0), output[0], j * batch_size, batch_size, image_size, shuffle_array);
			copy_mem(labels.get_data(0), target, j * batch_size, batch_size, category_count, shuffle_array);

			for (int k = 0; k < layer_count; k++)
				layer[k]->feedforward(output[k], output[k + 1]);

			err_now = ANN::evaluate(output[layer_count], target, category_count, batch_size);
			err += err_now;

			printf("batch %d/%d : %.4f\n", j + 1, train_batch_count, err_now);
			
			for (int k = (layer_count - 1); k >= 0; k--)
				layer[k]->backpropagation(output[k + 1], target, output[k]);
		}
		printf("epoch %d error: %.4f\t", i + 1, err / train_batch_count);
		
		// VALIDATION STAGE
		
		err = 0.0f;

		for (int j = 0; j < valid_batch_count; j++) {
			copy_mem(images.get_data(train_image_count), output[0], j * batch_size, batch_size, image_size, 0);
			copy_mem(labels.get_data(train_image_count), target, j * batch_size, batch_size, category_count, 0);
			
			for (int k = 0; k < layer_count; k++)
				layer[k]->test(output[k], output[k + 1], batch_size);

			err += ANN::evaluate(output[layer_count], target, category_count, batch_size);
		}
		printf("validation_error: %.4f\n", err / valid_batch_count);
	}

	for (int i = 0; i < layer_count + 1; i++)
		delete[] output[i];
	delete[] output;
	delete[] shuffle_array;
	delete[] target;
}

LabelArray ANN::test(const ImageArray &images) {
	LabelArray prediction(category_count);
	prediction.alloc(images.count());

	float** output = new float* [layer_count + 1];
	output[0] = init_mem(layer[0]->icrow * layer[0]->iccol * layer[0]->idepth * images.count());
	for (int i = 0; i < layer_count; i++)
		output[i + 1] = init_mem(layer[i]->ocrow * layer[i]->occol * layer[i]->odepth * images.count());

	copy_mem(images.get_data(0), output[0], 0, images.count(), images.item_size(), 0);

	for (int i = 0; i < layer_count; i++)
		layer[i]->test(output[i], output[i + 1], images.count());

	copy_mem(output[layer_count], prediction.data, 0, images.count(), category_count, 0);
	
	for (int i = 0; i < layer_count + 1; i++)
		delete[] output[i];
	delete[] output;

	return prediction;
}

float ANN::evaluate(const float* output, const float* target, int category_count, int cbatch) {
	float err = 0.0f;

	for (int i = 0; i < cbatch; i++)
		for (int j = 0; j < category_count; j++)
			err -= target[i * category_count + j] * log(output[i * category_count + j]);

	return err / cbatch;
}

void ANN::save_to(const char* file_path) {
	std::ofstream file;

	file.open(file_path, std::ios::binary);
	file.write((char*)&layer_count, sizeof(layer_count));
	file.write((char*)&batch_size, sizeof(batch_size));
	file.write((char*)&category_count, sizeof(category_count));

	for (int i = 0; i < layer_count; i++)
		layer[i]->save_to(file);

	file.close();
}

void ANN::load_from(const char* file_path) {
	std::ifstream file;
	int serial_id;

	file.open(file_path, std::ios::binary);

	if (!file.is_open()) {
		printf("The file \"%s\" couldn't open\n", file_path);
		exit(1);
	}

	file.read((char*)&layer_count, sizeof(layer_count));
	file.read((char*)&batch_size, sizeof(batch_size));
	file.read((char*)&category_count, sizeof(category_count));

	layer = new Layer * [layer_count];
	
	for (int i = 0; i < layer_count; i++) {
		file.read((char*)&serial_id, sizeof(int));
		switch (serial_id) {
		case BatchNorm::SERIALIZE_ID:	layer[i] = new BatchNorm; layer[i]->load_from(file); break;
		case Conv2D::SERIALIZE_ID:		layer[i] = new Conv2D; layer[i]->load_from(file); break;
		case Dense::SERIALIZE_ID:		layer[i] = new Dense; layer[i]->load_from(file); break;
		case MaxPool::SERIALIZE_ID:		layer[i] = new MaxPool; layer[i]->load_from(file); break;
		case Relu::SERIALIZE_ID:		layer[i] = new Relu; layer[i]->load_from(file); break;
		case Softmax::SERIALIZE_ID:		layer[i] = new Softmax; layer[i]->load_from(file); break;
		case Tanh::SERIALIZE_ID:		layer[i] = new Tanh; layer[i]->load_from(file); break;
		case LeakyRelu::SERIALIZE_ID:	layer[i] = new LeakyRelu; layer[i]->load_from(file); break;
		case Elu::SERIALIZE_ID:			layer[i] = new Elu; layer[i]->load_from(file); break;
		case Sigmoid::SERIALIZE_ID:		layer[i] = new Sigmoid; layer[i]->load_from(file); break;
		}
	}

	file.close();
}
