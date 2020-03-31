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
    data_points_num = 0;
}

void DBSCAN::register_parameters()
{
	watch_param("min_points", &min_points);
	watch_param("epsilon", &epsilon);
    watch_param("cluster_ids", &cluster_ids, &data_points_num);
    watch_param("data_points_num", &data_points_num);
}

DBSCAN::~DBSCAN()
{
    SG_FREE(cluster_ids);
}

bool DBSCAN::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	auto lhs = distance->get_lhs();
	ASSERT(lhs)

    data_points_num = lhs->get_num_vectors();
	ASSERT(data_points_num > 0)

	SG_FREE(cluster_ids);
	cluster_ids = SG_MALLOC(int16_t, data_points_num);
	SGVector<int16_t>::fill_vector(cluster_ids, data_points_num, UNCLASSIFIED);

	int16_t cluster_type = CORE_POINT;
	for (auto i : SG_PROGRESS(range(0, data_points_num)))
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

int16_t DBSCAN::expand_cluster(int32_t point_index, int16_t cluster_id)
{
	DynamicArray<int32_t> cluster_seeds = calculate_cluster(point_index);

	if (cluster_seeds.get_array_size() < min_points)
	{
		cluster_ids[point_index] = NOISE;
		return FAILURE;
	}
	else
	{
		int index_core_point = 0;
		for (int32_t i = 0; i < cluster_seeds.get_array_size(); ++i)
		{
			cluster_ids[cluster_seeds[i]] = cluster_id;
			if (cluster_seeds[i] == point_index)
			{
                index_core_point = i;
			}
		}

		cluster_seeds.delete_element(index_core_point);

		for (int32_t i = 0, n = cluster_seeds.get_array_size(); i < n; ++i)
		{
			DynamicArray<int32_t> cluster_neighbors =
			    calculate_cluster(cluster_seeds[i]);

			if (cluster_neighbors.get_array_size() >= min_points)
			{
				DynamicArray<int32_t> neighbor_indexes;

				for (int32_t j = 0; j < neighbor_indexes.get_array_size(); ++j)
				{
					int32_t neighbors_index = neighbor_indexes[j];
					if (cluster_ids[neighbors_index] == UNCLASSIFIED ||
					    cluster_ids[neighbors_index] == NOISE)
					{
						if (cluster_ids[neighbors_index] == UNCLASSIFIED)
						{
							cluster_seeds.push_back(neighbors_index);
							n = cluster_seeds.get_array_size();
						}
						cluster_ids[neighbors_index] = cluster_id;
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
	for (int32_t i = 0; i < data_points_num; ++i)
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

int32_t DBSCAN::get_min_points()
{
	return min_points;
}

float64_t DBSCAN::get_eps()
{
	return epsilon;
}

int32_t DBSCAN::get_data_points_num()
{
	return data_points_num;
}
