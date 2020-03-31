//
// Created by arkan0id on 31.03.20.
//

#ifndef SHOGUN_NEURALRADIALBASISLAYER_H
#define SHOGUN_NEURALRADIALBASISLAYER_H

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/neuralnets/NeuralLayer.h>

using namespace shogun;

class NeuralRadialBasisLayer : public NeuralLayer
{
public:
    /** default constructor */
    NeuralRadialBasisLayer();

    /** Constuctor
     *
     * @param num_neurons Number of neurons in this layer
     */
    NeuralRadialBasisLayer(int32_t num_neurons, Distance* d);

    virtual ~NeuralRadialBasisLayer() {}

    /** Computes the activations of the neurons in this layer, results should
     * be stored in m_activations. To be used only with non-input layers
     *
     * @param parameters Vector of size get_num_parameters(), contains the
     * parameters of the layer
     *
     * @param layers Array of layers that form the network that this layer is
     * being used with
     */
    virtual void compute_activations(SGVector<float64_t> parameters,
                                     const std::vector<std::shared_ptr<NeuralLayer>>& layers);

    /** Computes the gradients of the error with respect to this layer's
     * pre-activations. Results are stored in m_local_gradients.
     *
     * This is used by compute_gradients() and can be overriden to implement
     * layers with different activation functions
     *
     * @param targets a matrix of size num_neurons*batch_size. If is_output is
     * true, targets is the desired values for the layer's activations,
     * otherwise it's an empty matrix
     */
    virtual void compute_local_gradients(SGMatrix<float64_t> targets);

    /** Computes the error between the layer's current activations and the given
     * target activations. Should only be used with output layers
     *
     * @param targets desired values for the layer's activations, matrix of size
     * num_neurons*batch_size
     */
    virtual float64_t compute_error(SGMatrix<float64_t> targets);

    virtual const char* get_name() const { return "NeuralRadialBasisLayer"; }

private:
    void init();

protected:
    Distance* distance;
};


#endif // SHOGUN_NEURALRADIALBASISLAYER_H
