#pragma once
#include <fstream>
#include "alloc.h"

class ImageArray {
private:
	int width;
	int height;
	int channel;
	int image_count;
	int train_image_count;
	int valid_image_count;
	float valid_split;
	float* data;
public:
	ImageArray(int, int, int);
	~ImageArray();
	void load_from(const char*);
	void alloc(int);
	void split(float);
	int item_size() const;
	int count() const;
	int train_sample_count() const;
	int valid_sample_count() const;
	float* get_data(int) const;
	float* raw();
	static void batch_normalization(ImageArray&, ImageArray&);
	static void min_max_scaling(ImageArray&);
	friend class CNN;
};