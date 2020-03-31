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

NeuralRadialBasisLayer::NeuralRadialBasisLayer(int32_t num_neurons, Distance* distance_)
: NeuralLayer(num_neurons)
{
    init();
    distance = distance_;
}

void NeuralRadialBasisLayer::compute_activations(
    SGVector<float64_t> parameters,
    const std::vector<std::shared_ptr<NeuralLayer>>& layers)
{
//    NeuralLinearLayer::compute_activations(parameters, layers);
//
//    // to avoid exponentiating large numbers, the maximum activation is
//    // subtracted from all the activations and the computations are done in the
//    // log domain
//
//    float64_t max = m_activations.max_single();
//
//    for (int32_t j=0; j<m_batch_size; j++)
//    {
//        float64_t sum = 0;
//        for (int32_t i=0; i<m_num_neurons; i++)
//        {
//            sum += std::exp(m_activations[i + j * m_num_neurons] - max);
//        }
//        float64_t normalizer = std::log(sum);
//        for (int32_t k=0; k<m_num_neurons; k++)
//        {
//            m_activations[k + j * m_num_neurons] = std::exp(
//                m_activations[k + j * m_num_neurons] - max - normalizer);
//        }
//    }
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

float64_t NeuralRadialBasisLayer::compute_error(SGMatrix<float64_t> targets)
{
//    int32_t len = m_num_neurons*m_batch_size;
//    float64_t sum = 0;
//    for (int32_t i=0; i< len; i++)
//    {
//        // to prevent taking the log of a zero
//        if (m_activations[i]==0)
//            sum += targets[i] * std::log(1e-50);
//        else
//            sum += targets[i] * std::log(m_activations[i]);
//    }
//    return -1*sum/m_batch_size;
}

void NeuralRadialBasisLayer::init()
{
    distance = new EuclideanDistance();

    SG_ADD(&distance, "distance", "Radial basis layer distance");

//    SG_ADD_OPTIONS(
//        (machine_int_t*)&m_activation_function, "activation_function",
//        "Activation Function", ParameterProperties::NONE,
//        SG_OPTIONS(CMAF_IDENTITY, CMAF_LOGISTIC, CMAF_RECTIFIED_LINEAR));
}
