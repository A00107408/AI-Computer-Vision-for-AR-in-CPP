#pragma once

#include "neuron.h"
#include <stdint.h>
#include <vector>

static int layerId = 0;

class Layer
{

public:
	int currentLayerSize;
	std::vector<Neuron *> neurons;
	std::vector<double> layerOutputs;
	Layer(int, int);
	~Layer();
	std::vector<double> getLayerOutputs();
	int getSize();
};

