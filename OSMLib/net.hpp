#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#include "settings.hpp"

// type to use for neurons
#define OSML_TYPE_NEURON double
#if OSML_DTYPE_1
	#define OSML_TYPE_NEURON float
#endif

// type to use for calculations
#define OSML_TYPE_CALC double
#if OSML_DTYPE_2
	#define OSML_TYPE_CALC float
#endif

// pack structures?
#if OSML_PACK
	#pragma pack(1)
#endif

namespace osml
{

namespace ActFuncs {
	OSML_TYPE_CALC linear(OSML_TYPE_CALC x) {
		return x;
	}

	OSML_TYPE_CALC sigmoid(OSML_TYPE_CALC x) {
		return 1 / (1 + exp(-x));
	}

	OSML_TYPE_CALC bin(OSML_TYPE_CALC x) {
		return x > 0 ? 1 : 0;
	}

	OSML_TYPE_CALC tanh(OSML_TYPE_CALC x) {
		return std::tanh(x);
	}
	OSML_TYPE_CALC tanh_derivative(OSML_TYPE_CALC x) {
		return 1 - (std::tanh(x) * std::tanh(x));
	}

	OSML_TYPE_CALC relu(OSML_TYPE_CALC x) {
		return x > 0 ? x : 0;
	}
	OSML_TYPE_CALC relu_leaky(OSML_TYPE_CALC x) {
		return x > 0 ? x : 0.01 * x;
	}
	OSML_TYPE_CALC relu_parametric(OSML_TYPE_CALC x, OSML_TYPE_CALC param) {
		return x > 0 ? x : param * x;
	}

	OSML_TYPE_CALC elu(OSML_TYPE_CALC x) {
		return x > 0 ? x : exp(x) - 1;
	}
}


class Neuron {
public:
	OSML_TYPE_NEURON Weight;

	Neuron(OSML_TYPE_NEURON weight) : Weight(weight) {}

	OSML_TYPE_CALC calculate(
		const std::vector<OSML_TYPE_CALC>& inputs, 
		OSML_TYPE_NEURON bias = 0
	) {
		OSML_TYPE_CALC sum = 0;
		for (auto& i : inputs) sum += i;
		// activation function
		sum = ActFuncs::sigmoid(sum);
		return (sum * Weight) + bias;
	}
};

class Net {
public:
	Net() {}

	std::vector<std::vector<Neuron>> Layers {};

};

}

/*
Operations done by neurons:
1. Sum all inputs
2. Multiply by weight
3. Add bias
4. Apply activation function (turns the output into a value between 0 and 1)
*/