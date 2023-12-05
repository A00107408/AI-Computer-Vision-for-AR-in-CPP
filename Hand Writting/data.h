#pragma once

#ifndef _DATA_H
#define _DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"

class data
{
	std::vector<uint8_t> * feature_vector;
	uint8_t label;
	int enum_label; // A -> 1, B -> 2...

public:
	data();
	~data();
	void set_feature_vector(std::vector <uint8_t> *);
	void append_to_feature_vector(uint8_t);
	void set_label(uint8_t);
	void set_enumerated_label(int);

	int get_feature_vector_size();
	uint8_t get_label();
	uint8_t get_enumerated_label();

	std::vector<uint8_t>* get_feature_vector();
};

#endif // !_DATA_H
