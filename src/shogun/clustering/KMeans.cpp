/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Saurabh Mahindre,
 *          Sergey Lisitsyn, Evan Shelhamer, Soumyajit De, Fernando Iglesias,
 *          Bjoern Esser, parijat
 */

#include <shogun/base/progress.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

using namespace Eigen;
using namespace shogun;


namespace shogun
{

KMeans::KMeans():KMeansBase()
{
}

KMeans::KMeans(int32_t k_i, std::shared_ptr<Distance> d_i, bool use_kmpp_i):KMeansBase(k_i, std::move(d_i), use_kmpp_i)
{
}

KMeans::KMeans(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i):KMeansBase(k_i, std::move(d_i), centers_i)
{
}

KMeans::~KMeans()
{
}

void KMeans::Lloyd_KMeans(SGMatrix<float64_t> centers, int32_t num_centers)
{
	auto lhs = distance->get_lhs()->as<DenseFeatures<float64_t>>();

	int32_t lhs_size = lhs->get_num_vectors();

	SGVector<int32_t> cluster_assignments = SGVector<int32_t>(lhs_size);
	cluster_assignments.zero();

	/* Weights : Number of points in each cluster */
	SGVector<int64_t> weights_set(num_centers);
	weights_set.zero();
	/* Initially set all weights for zeroth cluster, Changes in assignement step
	 */
	weights_set[0] = lhs_size;

	distance->precompute_lhs();

	for (auto iter : SG_PROGRESS(range(max_iter)))
	{
		if (iter == max_iter - 1)
			io::warn(
				"KMeans clustering has reached maximum number of ( {} ) "
				"iterations without having converged. Terminating. ",
				iter);

		int32_t changed;
		auto rhs_mus =
			std::make_shared<DenseFeatures<float64_t>>(centers.clone());
		distance->replace_rhs(rhs_mus);

		auto change_centers_step = [this,
			                        &centers](ChangeCentersContext context) {
			if (fixed_centers)
			{
				SGVector<float64_t> vec =
					context.lhs->get_feature_vector(context.i);
				float64_t temp_min =
					1.0 / context.weights_set[context.min_cluster];

				/* mu_new = mu_old + (x - mu_old)/(w) */
				for (int32_t j = 0; j < context.dim; ++j)
				{
					centers(j, context.min_cluster) +=
						(vec[j] - centers(j, context.min_cluster)) * temp_min;
				}

				context.lhs->free_feature_vector(vec, context.i);

				/* mu_new = mu_old - (x - mu_old)/(w-1) */
				/* if weights_set(j)~=0 */
				if (context.weights_set[context.cluster_assignments_i] != 0)
				{
					float64_t temp_i =
						1.0 /
						context.weights_set[context.cluster_assignments_i];
					SGVector<float64_t> vec1 =
						context.lhs->get_feature_vector(context.i);

					for (int32_t j = 0; j < context.dim; ++j)
					{
						centers(j, context.cluster_assignments_i) -=
							(vec1[j] -
							 centers(j, context.cluster_assignments_i)) *
							temp_i;
					}
					context.lhs->free_feature_vector(vec1, context.i);
				}
				else
				{
					centers.get_column(context.cluster_assignments_i).zero();
				}
			}
		};

		std::tie(cluster_assignments, weights_set, changed) =
			compute_cluster_assignments(
			    num_centers, change_centers_step, cluster_assignments,
			    weights_set);

		if (changed == 0)
			break;

		/* Update Step : Calculate new means */
		if (!fixed_centers)
		{
			centers.zero();

			for (int32_t i = 0; i < lhs_size; i++)
			{
				int32_t cluster_i = cluster_assignments[i];

				auto vec = lhs->get_feature_vector(i);
				linalg::add_col_vec(centers, cluster_i, vec, centers);
				lhs->free_feature_vector(vec, i);
			}

			for (int32_t i = 0; i < num_centers; i++)
			{
				if (weights_set[i] != 0)
				{
					auto col = centers.get_column(i);
					linalg::scale(col, col, 1.0 / weights_set[i]);
				}
			}
		}

		observe<SGMatrix<float64_t>>(iter, "cluster_centers");

		if (iter % (max_iter / 10) == 0)
			io::info(
				"Iteration[{}/{}]: Assignment of {} patterns changed.", iter,
				max_iter, changed);
	}
}

bool KMeans::train_machine(std::shared_ptr<Features> data)
{
	initialize_training(data);
	auto rhs_cache = distance->get_rhs();
	Lloyd_KMeans(cluster_centers, k);
	compute_stds();
	distance->replace_rhs(rhs_cache);
	compute_cluster_variances();
	auto cluster_centres =
		std::make_shared<DenseFeatures<float64_t>>(cluster_centers);
	distance->replace_lhs(cluster_centres);
	return true;
}

}

