//
// Created by arkan0id on 30.03.20.
//

#ifndef SHOGUN_DBSCAN_H
#define SHOGUN_DBSCAN_H

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>


namespace shogun
{
    enum EClusterType
    {
        UNCLASSIFIED,
        CORE_POINT,
        BORDER_POINT,
        NOISE,
        SUCCESS,
        FAILURE
    };

    class DistanceMachine;

    class DBSCAN : public DistanceMachine
    {
    public:
        /** default constructor */
        DBSCAN();

        /** constructor
         *
         * @param min_points
         * @param eps
         * @param points
         * @param d distance
         */
        DBSCAN(uint32_t min_points, float eps, std::shared_ptr<Distance> d);

        virtual ~DBSCAN();

        MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

        /** get classifier type
         *
         * @return classifier type DBSCAN
         */
        virtual EMachineType get_classifier_type() { return CT_DBSCAN; }

        /** load distance machine from file
         *
         * @param srcfile file to load from
         * @return if loading was successful
         */
        virtual bool load(FILE* srcfile);

        /** save distance machine to file
         *
         * @param dstfile file to save to
         * @return if saving was successful
         */
        virtual bool save(FILE* dstfile);

        /** @return object name */
        virtual const char* get_name() const { return "DBSCAN"; }

        virtual bool train_require_labels() const
        {
            return false;
        }

    protected:
        /** Initialize training for DBSCAN algorithm */
        void initialize_training(const std::shared_ptr<Features>& data=NULL);

        /**
         * Init the model (register params)
         */
        void init();

    protected:
        uint32_t min_points;

        float64_t epsilon;

        SGMatrix<float64_t> points;

        uint32_t point_size;

        EClusterType* cluster_types;
    };
}

#endif // SHOGUN_DBSCAN_H
