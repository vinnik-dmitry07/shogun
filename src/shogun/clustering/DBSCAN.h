//
// Created by arkan0id on 30.03.20.
//

#ifndef _DBSCAN_H__
#define _DBSCAN_H__

#include <shogun/lib/config.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/machine/DistanceMachine.h>

#define FAILURE -3
#define NOISE -2
#define UNCLASSIFIED -1
#define SUCCESS 0
#define CORE_POINT 1
#define BORDER_POINT 2

namespace shogun
{
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
		DBSCAN(int32_t min_points, float64_t eps, std::shared_ptr<Distance> d);

		virtual ~DBSCAN();

		MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

		/** get classifier type
		 *
		 * @return classifier type DBSCAN
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_DBSCAN;
		}

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
		virtual const char* get_name() const
		{
			return "DBSCAN";
		}

		virtual bool train_require_labels() const
		{
			return false;
		}

        int32_t get_min_points();

        float64_t get_eps();

        int32_t get_data_points_num();
	protected:
		virtual bool train_machine(std::shared_ptr<Features> data = NULL);

        int16_t expand_cluster(int32_t point_index, int16_t cluster_id);

		DynamicArray<int32_t> calculate_cluster(int32_t point_index);

        int32_t data_points_num;

		int32_t min_points;

		float64_t epsilon;

        int16_t* cluster_ids;

	private:
		/** Initialize attributes */
		void init();

		/** Register all parameters (aka this class' attributes) */
		void register_parameters();


	};
} // namespace shogun

#endif // _DBSCAN_H__
