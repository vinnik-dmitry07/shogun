/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Pan Deng, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#ifndef _DIRECTORKERNEL_H___
#define _DIRECTORKERNEL_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class DirectorKernel: public Kernel
{
	public:
		/** default constructor
		 *
		 */
		DirectorKernel()
		: Kernel(), external_features(false)
		{
		}

		/**
		 */
		DirectorKernel(bool is_external_features)
		: Kernel(), external_features(is_external_features)
		{
		}

		/** constructor
		 *
		 */
		DirectorKernel(int32_t size, bool is_external_features)
		: Kernel(size), external_features(is_external_features)
		{
		}

		/** default constructor
		 *
		 */
		~DirectorKernel() override
		{
			cleanup();
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override
		{
			if (env()->get_num_threads()!=1)
			{
				io::warn("Enforcing to use only one thread due to restrictions of directors");
				env()->set_num_threads(1);
			}
			return Kernel::init(l, r);
		}

		/** set the current kernel normalizer
		 *
		 * @return if successful
		 */
		bool set_normalizer(std::shared_ptr<KernelNormalizer> normalizer) override
		{
			return Kernel::set_normalizer(normalizer);
		}

		/** obtain the current kernel normalizer
		 *
		 * @return the kernel normalizer
		 */
		virtual std::shared_ptr<KernelNormalizer> get_normalizer()
		{
			return Kernel::get_normalizer();
		}

		/** initialize the current kernel normalizer
		 *  @return if init was successful
		 */
		bool init_normalizer() override
		{
			return Kernel::init_normalizer();
		}

		/** clean up your kernel
		 *
		 * base method only removes lhs and rhs
		 * overload to add further cleanup but make sure Kernel::cleanup() is
		 * called
		 */
		void cleanup() override
		{
			Kernel::cleanup();
		}

		virtual float64_t kernel_function(int32_t idx_a, int32_t idx_b)
		{
			error("Kernel function of Director Kernel needs to be overridden.");
			return 0;
		}

		/**
		 * get column j
		 *
		 * @return the jth column of the kernel matrix
		 */
		SGVector<float64_t> get_kernel_col(int32_t j) override
		{
			return Kernel::get_kernel_col(j);
		}

		/**
		 * get row i
		 *
		 * @return the ith row of the kernel matrix
		 */
		virtual SGVector<float64_t> get_kernel_row(int32_t i)
		{
			return Kernel::get_kernel_row(i);
		}

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual int32_t get_num_vec_lhs()
		{
			return Kernel::get_num_vec_lhs();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual int32_t get_num_vec_rhs()
		{
			return Kernel::get_num_vec_rhs();
		}

		/** set number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual void set_num_vec_lhs(int32_t num)
		{
			num_lhs=num;
		}

		/** set number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual void set_num_vec_rhs(int32_t num)
		{
			num_rhs=num;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual bool has_features()
		{
			if (!external_features)
				return Kernel::has_features();
			else
				return true;
		}

		/** remove lhs and rhs from kernel */
		virtual void remove_lhs_and_rhs()
		{
			Kernel::remove_lhs_and_rhs();
		}

		/** remove lhs from kernel */
		void remove_lhs() override
		{
			Kernel::remove_lhs();
		}

		/** remove rhs from kernel */
		void remove_rhs() override
		{
			Kernel::remove_rhs();
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type DIRECTOR
		 */
		EKernelType get_kernel_type() override { return K_DIRECTOR; }

		 /** return what type of features kernel can deal with
		  *
		  * @return feature type ANY
		  */
		EFeatureType get_feature_type() override { return F_ANY; }

		 /** return what class of features kernel can deal with
		  *
		  * @return feature class ANY
		  */
		EFeatureClass get_feature_class() override { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Director
		 */
		virtual const char* get_name() const { return "DirectorKernel"; }

		/** for optimizable kernels, i.e. kernels where the weight
		 * vector can be computed explicitly (if it fits into memory)
		 */
		virtual void clear_normal()
		{
			Kernel::clear_normal();
		}

		/** add vector*factor to 'virtual' normal vector
		 *
		 * @param vector_idx index
		 * @param weight weight
		 */
		virtual void add_to_normal(int32_t vector_idx, float64_t weight)
		{
			Kernel::add_to_normal(vector_idx, weight);
		}

		/** set optimization type
		 *
		 * @param t optimization type to set
		 */
		virtual void set_optimization_type(EOptimizationType t)
		{
			Kernel::set_optimization_type(t);
		}

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t *weights)
		{
			return Kernel::init_optimization(count, IDX, weights);
		}

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization()
		{
			return Kernel::delete_optimization();
		}

		/** compute optimized
		 *
		 * @param vector_idx index to compute
		 * @return optimized value at given index
		 */
		virtual float64_t compute_optimized(int32_t vector_idx)
		{
			return Kernel::compute_optimized(vector_idx);
		}

		/** computes output for a batch of examples in an optimized fashion
		 * (favorable if kernel supports it, i.e. has KP_BATCHEVALUATION.  to
		 * the outputvector target (of length num_vec elements) the output for
		 * the examples enumerated in vec_idx are added. therefore make sure
		 * that it is initialized with ZERO. the following num_suppvec, IDX,
		 * alphas arguments are the number of support vectors, their indices
		 * and weights
		 */
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0)
		{
			Kernel::compute_batch(num_vec, vec_idx, target, num_suppvec, IDX, alphas, factor);
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		virtual int32_t get_num_subkernels()
		{
			return Kernel::get_num_subkernels();
		}

		/** compute by subkernel
		 *
		 * @param vector_idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		virtual void compute_by_subkernel(
			int32_t vector_idx, float64_t * subkernel_contrib)
		{
			Kernel::compute_by_subkernel(vector_idx, subkernel_contrib);
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		virtual const float64_t* get_subkernel_weights(int32_t& num_weights)
		{
			return Kernel::get_subkernel_weights(num_weights);
		}

		/** set subkernel weights
		 *
		 * @param weights new subkernel weights
		 */
		virtual void set_subkernel_weights(SGVector<float64_t> weights)
		{
			Kernel::set_subkernel_weights(weights);
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_pre() noexcept(false)
		{
			Kernel::load_serializable_pre();
		}

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_post() noexcept(false)
		{
			Kernel::load_serializable_post();
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_pre() noexcept(false)
		{
			Kernel::save_serializable_pre();
		}

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_post() noexcept(false)
		{
			Kernel::save_serializable_post();
		}

protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			return kernel_function(idx_a, idx_b);
		}

		virtual void register_params()
		{
			Kernel::register_params();
		}

	protected:
		/* is external features */
		bool external_features;
};
}
#endif /* USE_SWIG_DIRECTORS */
#endif /* _DIRECTORKERNEL_H__ */
