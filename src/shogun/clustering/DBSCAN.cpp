#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

DBSCAN::DBSCAN() : DistanceMachine()
{
	init();
	register_parameters();
}

DBSCAN::DBSCAN(int32_t min_points_, float64_t eps_, std::shared_ptr<Distance> d)
    : DistanceMachine()
{
	init();
	min_points = min_points_;
	eps = eps_;
	set_distance(std::move(d));
	register_parameters();
}

void DBSCAN::init()
{
	min_points = 4;
	epsilon = 0.75 * 0.75;
	cluster_types = NULL;
	cluster_types_len = 0;
}

void DBSCAN::register_parameters()
{
	watch_param("min_points", &min_points);
	watch_param("epsilon", &epsilon);
	watch_param("cluster_types", &cluster_types, &cluster_types_len);
}

DBSCAN::~DBSCAN()
{
	SG_FREE(merge_distance);
	SG_FREE(assignment);
	SG_FREE(pairs);
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

	SG_FREE(cluster_types);
	cluster_types = SG_MALLOC(EClusterType, num);
	cluster_types_len = num;
	SGVector<EClusterType>::fill_vector(cluster_types, num, UNCLASSIFIED);

	float64_t* distances = SG_MALLOC(float64_t, num_pairs);

	int32_t cluster_id = 1;
	for (auto i : SG_PROGRESS(range(0, num)))
	{
		if (cluster_types[i] == UNCLASSIFIED)
		{
			if (expandCluster(i, cluster_id) != FAILURE)
			{
				cluster_id += 1;
			}
		}
	}

	return true;
}

int DBSCAN::expandCluster(uint32_t point_index, int cluster_id)
{
	DynamicArray<int32_t> cluster_seeds = calculate_cluster(point_index);

	if (cluster_seeds.size() < min_points)
	{
		cluster_types[point_index] = NOISE;
		return FAILURE;
	}
	else
	{
		int index = 0;
		int indexCorePoint = 0;
		vector<int>::iterator iterSeeds;
		for (int32_t iterSeeds = 0; iterSeeds != cluster_seeds.get_dim1();
		     ++iterSeeds)
		{
			cluster_types[cluster_seeds[iterSeeds]] = cluster_id;
			if (m_points.at(cluster_seeds[iterSeeds]).x == point.x &&
			    m_points.at(cluster_seeds[iterSeeds]).y == point.y &&
			    m_points.at(cluster_seeds[iterSeeds]).z == point.z)
			{
				indexCorePoint = index;
			}
			++index;
		}
		cluster_seeds.erase(cluster_seeds.begin() + indexCorePoint);

		for (vector<int>::size_type i = 0, n = cluster_seeds.size(); i < n; ++i)
		{
			vector<int> clusterNeighors =
			    calculateCluster(m_points.at(cluster_seeds[i]));

			if (clusterNeighors.size() >= m_minPoints)
			{
				vector<int>::iterator iterNeighors;
				for (iterNeighors = clusterNeighors.begin();
				     iterNeighors != clusterNeighors.end(); ++iterNeighors)
				{
					if (m_points.at(*iterNeighors).cluster_id == UNCLASSIFIED ||
					    m_points.at(*iterNeighors).clusterID == NOISE)
					{
						if (m_points.at(*iterNeighors).cluster_id ==
						    UNCLASSIFIED)
						{
							cluster_seeds.push_back(*iterNeighors);
							n = cluster_seeds.size();
						}
						m_points.at(*iterNeighors).cluster_id = cluster_id;
					}
				}
			}
		}

		return SUCCESS;
	}
}

DynamicArray<int32_t> DBSCAN::calculate_cluster(uint32_t point_index)
{
	DynamicArray<int32_t> cluster_index;
	for (int32_t other_index = 0; other_index < num; ++other_index)
	{
		if (distance->distance(point_index, other_index) <= epsilon)
		{
			cluster_index.push_back(other_index);
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

int32_t DBSCAN::get_merges()
{
	return merges;
}

SGVector<int32_t> DBSCAN::get_assignment()
{
	return SGVector<int32_t>(assignment, table_size, false);
}

SGVector<float64_t> DBSCAN::get_merge_distances()
{
	return SGVector<float64_t>(merge_distance, merges, false);
}

SGMatrix<int32_t> DBSCAN::get_cluster_pairs()
{
	return SGMatrix<int32_t>(pairs, 2, merges, false);
}
