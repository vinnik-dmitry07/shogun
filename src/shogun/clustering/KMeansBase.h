/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Heiko Strathmann
 */

#ifndef _KMEANSBASE_H__
#define _KMEANSBASE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>
#include <shogun/mathematics/RandomMixin.h>


namespace shogun
{
class DistanceMachine;

/**
  Base Class for different KMeans clustering implementations.
  */
class KMeansBase : public RandomMixin<DistanceMachine>
{
	public:
		/** default constructor */
		KMeansBase();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp Set to true for using KMeans++ (default false)
		 */
		KMeansBase(int32_t k, std::shared_ptr<Distance> d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		*/
		KMeansBase(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i);
		
		virtual ~KMeansBase();


		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** get classifier type
		 *
		 * @return classifier type KMEANS
		 */
		virtual EMachineType get_classifier_type() { return CT_KMEANS; }

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

		/** get centers
		 *
		 * @return cluster centers or empty matrix if no radiuses are there (not trained yet)
		 */
		SGMatrix<float64_t> get_cluster_centers() const;

		/** @return object name */
		virtual const char* get_name() const { return "KMeansBase"; }

		/** set the initial cluster centers
		 *
		 * @param centers matrix with cluster centers (k colums, dim rows)
		 */
		virtual void set_initial_centers(SGMatrix<float64_t> centers);

		virtual bool train_require_labels() const
		{
			return false;
		}

	protected:
		/** Initialize training for KMeans algorithms */
		void initialize_training(const std::shared_ptr<Features>& data=NULL);

		/** K-Means++ algorithm to initialize cluster centers
		*
		* @return initial cluster centers: matrix (k columns, dim rows)
		*/
		SGMatrix<float64_t> kmeanspp();

		/**
		 * Init the model (register params)
		 */
		void init();

		/** Algorithm to initialize random cluster centers
		*
		* @return initial cluster centers: matrix (k columns, dim rows)
		*/
		void set_random_centers();

		void compute_cluster_variances();

		void compute_stds();

        struct ChangeCentersContext
        {
            std::shared_ptr<DenseFeatures<float64_t>> lhs;
            int32_t i, dim, cluster_assignments_i, min_cluster;
            SGVector<int64_t> weights_set;
        };

        /** Matches points and clusters
        * @param change_centers optional coroutine to change centers in
        * Lloyd Kmeans
        * @return A tuple of: \n
        * 1) an assignments vector whose index is a point number and value is
        * a point cluster \n
        * 2) a weights vector whose index is a cluster number and value is
        * a number of points belong to this cluster \n
        * 3) a number of changed assignments
        */
        std::tuple<SGVector<int32_t>, SGVector<int64_t>, int32_t>
        compute_cluster_assignments(
            int32_t num_centers,
            std::function<void(ChangeCentersContext)> change_centers = nullptr,
            SGVector<int32_t> cluster_assignments = SGVector<int32_t>(),
            SGVector<int64_t> weights_set = SGVector<int64_t>());
	protected:
		/** Maximum number of iterations */
		int32_t max_iter;

		/** If cluster centers are to be kept fixed */
		bool fixed_centers;

		/** The k parameter in KMeans */
		int32_t k;

		/** Number of dimensions */
		int32_t dimensions;

		/** Radi of the clusters (size k) */
		SGVector<float64_t> R;

        /** Std of the clusters (size k) */
        SGMatrix<float64_t> stds;

		/** Initial centers supplied */
		SGMatrix<float64_t> initial_centers;

		/** Flag to check if kmeans++ has to be used */
		bool use_kmeanspp;

		/** Cluster centers */
		SGMatrix<float64_t> cluster_centers;
};
}
#endif

