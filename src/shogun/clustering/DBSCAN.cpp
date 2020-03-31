#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/clustering/DBSCAN.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

DBSCAN::DBSCAN() : DistanceMachine()
{
	init();
	register_parameters();
}

DBSCAN::DBSCAN(
    int32_t min_points_, float64_t epsilon_, std::shared_ptr<Distance> d)
    : DistanceMachine()
{
	init();
	min_points = min_points_;
	epsilon = epsilon_;
	set_distance(std::move(d));
	register_parameters();
}

void DBSCAN::init()
{
	min_points = 4;
	epsilon = 0.75 * 0.75;
	cluster_ids = NULL;
	cluster_ids_len = 0;
}

void DBSCAN::register_parameters()
{
	watch_param("min_points", &min_points);
	watch_param("epsilon", &epsilon);
	watch_param("cluster_ids", &cluster_ids, &cluster_ids_len);
}

DBSCAN::~DBSCAN()
{
	//	SG_FREE(merge_distance);
	//	SG_FREE(assignment);
	//	SG_FREE(pairs);
}

bool DBSCAN::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	auto lhs = distance->get_lhs();
	ASSERT(lhs)

	int32_t num = lhs->get_num_vectors();
	ASSERT(num > 0)

	SG_FREE(cluster_ids);
	cluster_ids = SG_MALLOC(int16_t, num);
	cluster_ids_len = num;
	SGVector<int16_t>::fill_vector(cluster_ids, num, UNCLASSIFIED);

	int16_t cluster_type = CORE_POINT;
	for (auto i : SG_PROGRESS(range(0, num)))
	{
		if (cluster_ids[i] == UNCLASSIFIED)
		{
			if (expand_cluster(i, cluster_type) != FAILURE)
			{
				cluster_type += 1;
			}
		}
	}

	return true;
}

int16_t DBSCAN::expand_cluster(int32_t point_index, int16_t cluster_type)
{
	DynamicArray<int32_t> cluster_seeds = calculate_cluster(point_index);

	if (cluster_seeds.get_array_size() < min_points)
	{
		cluster_ids[point_index] = NOISE;
		return FAILURE;
	}
	else
	{
		int index = 0;
		int index_core_point = 0;
		DynamicArray<int32_t> iterSeeds;
		for (int32_t i = 0; i < cluster_seeds.get_array_size(); ++i)
		{
			cluster_ids[cluster_seeds[i]] = cluster_type;
			if (points[cluster_seeds[i]] == points[point_index])
			{
                index_core_point = index;
			}
			++index;
		}

		cluster_seeds.delete_element(index_core_point);

		for (int32_t i = 0, n = cluster_seeds.get_array_size(); i < n; ++i)
		{
			DynamicArray<int32_t> cluster_neighbors =
			    calculate_cluster(cluster_seeds[i]);

			if (cluster_neighbors.get_array_size() >= min_points)
			{
				DynamicArray<int32_t> neighbor_indexes;

				// TODO modern traverse list
				for (int32_t k = 0; k < neighbor_indexes.get_array_size(); ++k)
				{
					int32_t neighbors_index = neighbor_indexes[k];
					if (cluster_ids[neighbors_index] == UNCLASSIFIED ||
					    cluster_ids[neighbors_index] == NOISE)
					{
						if (cluster_ids[neighbors_index] == UNCLASSIFIED)
						{
							cluster_seeds.push_back(neighbors_index);
							n = cluster_seeds.get_array_size();
						}
						cluster_ids[neighbors_index] = cluster_type;
					}
				}
			}
		}

		return SUCCESS;
	}
}

DynamicArray<int32_t> DBSCAN::calculate_cluster(int32_t point_index)
{
	DynamicArray<int32_t> cluster_index;
	for (int32_t i = 0; i < points.size(); ++i)
	{
		if (distance->distance(point_index, i) <= epsilon)
		{
			cluster_index.push_back(i);
		}
	}
	return cluster_index;
}

bool DBSCAN::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool DBSCAN::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGVector<int32_t> DBSCAN::get_min_points()
{
	return min_points;
}

SGVector<float64_t> DBSCAN::get_eps()
{
	return epsilon;
}
