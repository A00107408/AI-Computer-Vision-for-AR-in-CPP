#pragma once

#include <cmath>
#include <vector>

class Neuron
{
public:

	double output;
	double delta;
	std::vector<double> weights;

	double preActivation;
	double activeatedOutput;
	double outputDeriative;
	double error;
	double alpha; //used in activation functions

	Neuron(int, int);
	~Neuron();
	void initializeWeights(int);
	void setError(double);
	void setWeight(double, int);
	double calculatePreActivation(std::vector<double>);
	double calculateOutputDerivate();
	double sigmoid();
	double relu();
	double leakyRelu();
	double inverseSqrtRelu();
	double getOutput();
	double getOutputDerivative();
	double getError();
	std::vector<double> getWeights();
};

