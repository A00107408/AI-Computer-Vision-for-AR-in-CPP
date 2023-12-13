#include "Network.h"
#include "layer.h"
#include "data_handler.h"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate)
{
	for (int i = 0; i < spec.size(); i++)
	{
		if (i == 0)
		{
			layers.push_back(new Layer(inputSize, spec.at(i)));
		}
		else
		{
			layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i)));
		}
		layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), numClasses));
		this->learningRate = learningRate;
	}
}

Network::~Network() {}

double Network::activate(std::vector<double> weights, std::vector<double> input)
{
	double activation = weights.back(); // bias term

	for (int i = 0; i < weights.size() - 1; i++)
	{
		activation += weights[i] * input[i];
	}
	return activation;
}

double Network::transfer(double activation)
{
	return 1.0 / (1.0 + exp(-activation));
}

double Network::transferDerivative(double output)
{
	return output * (1 - output);
}

std::vector<double> Network::fprop(data* data)
{
	std::vector<double> inputs = *data->get_normalized_feature_vector(); 
	for (int i = 0; i < layers.size(); i++)
	{
		Layer* layer = layers.at(i);
		std::vector<double> newInputs;
		for (Neuron* n : layer->neurons)
		{
			double activation = this->activate(n->weights, inputs);
			n->output = this->transfer(activation);
			newInputs.push_back(n->output);
		}
		inputs = newInputs;
	}
	return inputs; //output layer outputs
}

void Network::bprop(data* data)
{
	for (int i = layers.size() - 1; i >= 0; i--)
	{
		Layer* layer = layers.at(i);
		std::vector<double> errors;
		if (i != layers.size() - 1)
		{
			for (int j = 0; j < layer->neurons.size(); j++)
			{
				double error = 0.0;
				for (Neuron* n : layers.at(i + 1)->neurons)
				{
					error += (n->weights.at(j) * n->delta);
				}
				errors.push_back(error);
			}
		}
		else
		{
			for (int j = 0; j < layer->neurons.size(); j++)
			{
				Neuron* n = layer->neurons.at(j);
				errors.push_back((double)data->get_class_vector()->at(j) - n->output); //expected - actual
			}
		}
		for (int j = 0; j < layer->neurons.size(); j++)
		{
			Neuron* n = layer->neurons.at(j);
			n->delta = errors.at(j) * this->transferDerivative(n->output); //gradient / derivative part of back proppagation.
		}
	}
}

void Network::updateWeights(data* data)
{
	std::vector<double> inputs = *data->get_normalized_feature_vector();
	for (int i = 0; i < layers.size(); i++)
	{
		if (i != 0)
		{
			for (Neuron* n : layers.at(i - 1)->neurons)
			{
				inputs.push_back(n->output);
			}
		}
		for (Neuron* n : layers.at(i)->neurons)
		{
			for (int j = 0; j < inputs.size(); j++)
			{
				n->weights.at(j) += this->learningRate * n->delta * inputs.at(j);
			}
			n->weights.back() += this->learningRate * n->delta;
		}
		inputs.clear();
	}
}

int Network::predict(data* data)
{
	std::vector<double> outputs = fprop(data);
	return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int numEpochs)
{
	for (int i = 0; i < numEpochs; i++)
	{
		double sumError = 0.0;
		for (data *data : *this->training_data)
		{
			std::vector<double> outputs = fprop(data);
			std::vector<int> expected = *data->get_class_vector(); //check in vid
			double tempErrorSum = 0.0;
			for (int j = 0; j < outputs.size(); j++)
			{
				tempErrorSum += pow((double)expected.at(j) - outputs.at(j), 2);
			}
			sumError += tempErrorSum;
			bprop(data);
			updateWeights(data);
		}
		printf("Iteration: %d \t Error=%.4f\n", i, sumError);
	}
}

double Network::test()
{
	double numCorrect = 0.0;
	double count = 0.0;
	for (data *data : *this->test_data)
	{
		count++;
		int index = predict(data);
		if(data->get_class_vector()->at(index) == 1) numCorrect++;
	}
	testPerformance = (numCorrect / count);
	return testPerformance;
}

void Network::validate()
{
	double numCorrect = 0.0;
	double count = 0.0;
	for (data* data : *this->validation_data)
	{
		count++;
		int index = predict(data);
		if (data->get_class_vector()->at(index) == 1) numCorrect++;
	}
	printf("Validation performance: %.4f\n", numCorrect / count);
}

int main() {
	
	data_handler* dh = new data_handler();

	//Uncomment to read MNIST binary files.
	dh->read_feature_vector("train-images.idx3-ubyte");
	dh->read_feature_labels("train-labels.idx1-ubyte");
	dh->split_data();
	dh->count_classes();

	//Uncomment to read iris.csv
	//dh->read_csv("iris.data", ",");

	dh->split_data();
	std::vector<int> hiddenLayers = { 10 };
	//auto lambda = [&]() {
		Network* net = new Network(
			hiddenLayers,
			dh->get_training_data()->at(0)->get_normalized_feature_vector()->size(),
			dh->get_class_counts(),
			0.25);
		//net->set_training_data(dh->get_training_data());
		//net->set_test_data(dh->get_test_data());
		//net->set_validation_data(dh->get_validation_data());
		//net->train(15);
		//net->validate();
		printf("Test Performance: %.3f\n", net->test());
	//};
	//lambda();
}


