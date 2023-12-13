#pragma once

#include "data.h"
#include "Neuron.h"
#include "layer.h"
#include "Hiddenlayer.h"
#include "Inputlayer.h"
#include "Outputlayer.h"
#include "common.h"

class Network : public common_data
{

private:
	InputLayer *inputLayer;
	OutputLayer *outputLayer;
	std::vector<HiddenLayer *> hiddenLayers;
	double eta; //learning rate

public:
	std::vector<Layer*> layers;
	double learningRate;
	double testPerformance;
	Network(std::vector<int> spec, int, int, double);
	~Network();

	std::vector<double> fprop(data *data);
	double activate(std::vector<double>, std::vector<double>); //dot product
	double transfer(double);
	double transferDerivative(double); //used for backprop
	void bprop(data *data);
	void updateWeights(data *data);
	int predict(data* data); //return the index of the maximu value in 
	void train(int);
	double test();
	void validate();
};

