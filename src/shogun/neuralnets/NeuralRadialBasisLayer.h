//
// Created by arkan0id on 31.03.20.
//

#ifndef _NEURALRADIALBASISLAYER_H__
#define _NEURALRADIALBASISLAYER_H__

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
     * @param d Distance
     */
    NeuralRadialBasisLayer(int32_t num_neurons, std::shared_ptr<Distance> d);

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
     * @param targets a matrix of size num_neurons*batch_size. If is_output is
     * true, targets is the desired values for the layer's activations,
     * otherwise it's an empty matrix
     */
    virtual void compute_local_gradients(SGMatrix<float64_t> targets);

    virtual const char* get_name() const { return "NeuralRadialBasisLayer"; }

private:
    void init();

protected:
    std::shared_ptr<Distance> distance;
};


#endif // _NEURALRADIALBASISLAYER_H__
