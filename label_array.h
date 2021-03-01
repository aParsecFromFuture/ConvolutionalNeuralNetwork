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
	void alloc(int);
	int* simplify();
	float* get_data(int) const;
	float* raw();
	int get_category_count() const;
	int get_label_count() const;
	friend class CNN;
};