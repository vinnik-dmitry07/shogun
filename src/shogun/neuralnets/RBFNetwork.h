//
// Created by arkan0id on 31.03.20.
//

#ifndef _RBFNETWORK_H__
#define _RBFNETWORK_H__


#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/neuralnets/NeuralRadialBasisLayer.h>

namespace shogun
{
    class RBFNetwork : public NeuralNetwork
    {
    public:
        /** default constructor */
        RBFNetwork();

        /** Constructor
         *
         * @param num_inputs Number of inputs
         */
        RBFNetwork(int32_t num_inputs, int32_t num_hidden, float64_t sigma = 0.01);

        /** Trains RBFNetwork
         *
         * @param data Training examples
         *
         * @return accuracy
         */
        virtual float64_t train(std::shared_ptr<Features> data);

        virtual ~RBFNetwork() {}

    private:
        void init();

    protected:
        int32_t num_hidden;
    };
}

#endif // _RBFNETWORK_H__
