//
// Created by arkan0id on 31.03.20.
//

#include "RBFNetwork.h"
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>

using namespace shogun;

RBFNetwork::RBFNetwork() : NeuralNetwork()
{
    init();
}

RBFNetwork::RBFNetwork(int32_t num_inputs, int32_t num_hidden, float64_t sigma) : NeuralNetwork()
{
    init();

    std::vector<std::shared_ptr<NeuralLayer>> layers;
    layers.push_back(std::make_shared<NeuralInputLayer>(num_inputs));
    auto distance = std::make_shared<EuclideanDistance>();
    layers.push_back(std::make_shared<NeuralRadialBasisLayer>(num_hidden, distance));
    layers.push_back(std::make_shared<NeuralLinearLayer>(1));

    set_layers(layers);

    quick_connect();

    initialize_neural_network(sigma);
}


float64_t RBFNetwork::train(std::shared_ptr<Features> data)
{

}

void RBFNetwork::init()
{
    num_hidden = 3;
}