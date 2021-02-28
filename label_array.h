#pragma once
#include <fstream>
#include "alloc.h"

class LabelArray {
private:
	int category_count;
	int label_count;
	float* data;
public:
	LabelArray(int);
	~LabelArray();
	void load_from(const char*);
	float* get_data(int) const ;
	friend class CNN;
};