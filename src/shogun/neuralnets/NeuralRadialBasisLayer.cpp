//
// Created by arkan0id on 31.03.20.
//

#include "NeuralRadialBasisLayer.h"
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/NormalDistribution.h>

using namespace shogun;

NeuralRadialBasisLayer::NeuralRadialBasisLayer() : NeuralLayer()
{
    init();
}

NeuralRadialBasisLayer::NeuralRadialBasisLayer(int32_t num_neurons, std::shared_ptr<Distance> distance_)
: NeuralLayer(num_neurons)
{
    init();
    distance = std::move(distance_);
}

void NeuralRadialBasisLayer::compute_activations(
    SGVector<float64_t> parameters,
    const std::vector<std::shared_ptr<NeuralLayer>>& layers)
{
    return exp(gamma * distance(data_point, centroid))
}

void NeuralRadialBasisLayer::compute_local_gradients(SGMatrix<float64_t> targets)
{
//    if (targets.num_rows == 0)
//        error("Cannot be used as a hidden layer");
//
//    int32_t len = m_num_neurons*m_batch_size;
//    for (int32_t i=0; i< len; i++)
//    {
//        m_local_gradients[i] = (m_activations[i]-targets[i])/m_batch_size;
//    }
}


void NeuralRadialBasisLayer::init()
{
    SG_ADD(&distance, "distance", "Radial basis layer distance");

//    SG_ADD_OPTIONS(
//        (machine_int_t*)&m_activation_function, "activation_function",
//        "Activation Function", ParameterProperties::NONE,
//        SG_OPTIONS(CMAF_IDENTITY, CMAF_LOGISTIC, CMAF_RECTIFIED_LINEAR));
}
