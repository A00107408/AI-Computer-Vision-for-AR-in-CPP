#pragma once

#include <vector>
#include "data.h"

class knn
{
	int k;
	std::vector<data *> * neighbors;
	std::vector<data *> * training_data;
	std::vector<data *> * test_data;
	std::vector<data *> * validation_data;

public:
	knn(int);
	knn();
	~knn();

	void find_knearest(data *query_point);
	
	void set_training_data(std::vector<data *> *vect);
	void set_test_data(std::vector<data *> *vect);
	void set_validation_data(std::vector<data *> *vect);
	void set_k(int val);

	int predict();
	double calculate_distance(data* query_point, data* input);
	double validate_performane();
	double test_performane();
};

