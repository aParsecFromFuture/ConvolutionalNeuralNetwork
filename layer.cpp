#include "layer.h"

const float EULER = 2.7182818f;
const float EPSILON = 1e-5;

Layer::Layer() {
	this->icrow = 0;
	this->iccol = 0;
	this->idepth = 0;

	this->icr = 0;
	this->idcr = 0;

	this->ocrow = 0;
	this->occol = 0;
	this->odepth = 0;

	this->ocr = 0;
	this->odcr = 0;

	this->cbatch = 0;
	this->lr = 0.0f;
	this->momentum = 0.0f;
}

void Layer::init(int icrow, int iccol, int idepth, int ocrow, int occol, int odepth, int cbatch) {
	this->icrow = icrow;
	this->iccol = iccol;
	this->idepth = idepth;

	this->icr = icrow * iccol;
	this->idcr = icrow * iccol * idepth;

	this->ocrow = ocrow;
	this->occol = occol;
	this->odepth = odepth;

	this->ocr = ocrow * occol;
	this->odcr = ocrow * occol * odepth;

	this->cbatch = cbatch;
}

void Layer::save_to(std::ofstream &file) {
	file.write((char*)&icrow, sizeof(icrow));
	file.write((char*)&iccol, sizeof(iccol));
	file.write((char*)&idepth, sizeof(idepth));
	file.write((char*)&icr, sizeof(icr));
	file.write((char*)&idcr, sizeof(idcr));
	file.write((char*)&ocrow, sizeof(ocrow));
	file.write((char*)&occol, sizeof(occol));
	file.write((char*)&odepth, sizeof(odepth));
	file.write((char*)&ocr, sizeof(ocr));
	file.write((char*)&odcr, sizeof(odcr));
	file.write((char*)&cbatch, sizeof(cbatch));
}

void Layer::load_from(std::ifstream &file) {
	file.read((char*)&icrow, sizeof(icrow));
	file.read((char*)&iccol, sizeof(iccol));
	file.read((char*)&idepth, sizeof(idepth));
	file.read((char*)&icr, sizeof(icr));
	file.read((char*)&idcr, sizeof(idcr));
	file.read((char*)&ocrow, sizeof(ocrow));
	file.read((char*)&occol, sizeof(occol));
	file.read((char*)&odepth, sizeof(odepth));
	file.read((char*)&ocr, sizeof(ocr));
	file.read((char*)&odcr, sizeof(odcr));
	file.read((char*)&cbatch, sizeof(cbatch));
}