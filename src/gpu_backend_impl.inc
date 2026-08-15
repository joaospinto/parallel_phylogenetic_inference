#if !defined(PARALLEL_PHYLOGENETICS_GPU_NAMESPACE) ||                          \
    !defined(PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE) ||                     \
    !defined(PARALLEL_PHYLOGENETICS_GPU_NAME) ||                               \
    !defined(PARALLEL_PHYLOGENETICS_GPU_HAS_DEVICE_SELECTION)
#error "gpu_backend_impl.inc requires a complete backend wrapper"
#endif

// CUDA and ROCm accept an explicit device index, whereas Metal exposes the
// process-selected device. This is the only application-layer API difference;
// all reservation and prepared-inference forwarding remains shared below.
#if PARALLEL_PHYLOGENETICS_GPU_HAS_DEVICE_SELECTION
#define PARALLEL_PHYLOGENETICS_DEVICE_PARAMETER int device
#define PARALLEL_PHYLOGENETICS_DEVICE_ARGUMENT device
#define PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(...) __VA_ARGS__, int device
#define PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(...) __VA_ARGS__, device
#else
#define PARALLEL_PHYLOGENETICS_DEVICE_PARAMETER
#define PARALLEL_PHYLOGENETICS_DEVICE_ARGUMENT
#define PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(...) __VA_ARGS__
#define PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(...) __VA_ARGS__
#endif

#include <stdexcept>
#include <utility>

#include "src/accelerator_internal.h"
namespace parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE {

struct Workspace::Impl
    : internal::WorkspaceStorage<
          tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::Workspace> {};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() {
  return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::Available();
}

std::string DeviceDescription(PARALLEL_PHYLOGENETICS_DEVICE_PARAMETER) {
  return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::DeviceDescription(
      PARALLEL_PHYLOGENETICS_DEVICE_ARGUMENT);
}

void Workspace::Reserve(PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(
    AlignmentModelView model, std::size_t site_batch_capacity)) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kLikelihood,
      [&](auto &backend_workspace, std::size_t batch) {
        backend_workspace.ReserveCategorical(
            PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(model.plan, 4, batch, 16,
                                                     model.observation_nodes));
      });
}

void Workspace::ReserveMaximum(PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(
    AlignmentModelView model, std::size_t site_batch_capacity)) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMaximum,
      [&](auto &backend_workspace, std::size_t batch) {
        backend_workspace.ReserveCategoricalMaximum(
            PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(model.plan, 4, batch, 16,
                                                     model.observation_nodes));
      });
}

void Workspace::ReserveSampling(PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(
    AlignmentModelView model, std::size_t site_batch_capacity)) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kSampling,
      [&](auto &backend_workspace, std::size_t batch) {
        backend_workspace.ReserveCategoricalSampling(
            PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(model.plan, 4, batch, 16,
                                                     model.observation_nodes));
      });
  if (storage.rate_categories > 1) {
    storage.sampling_likelihood_tree_hmms.resize(storage.rate_categories);
    for (auto &backend_workspace : storage.sampling_likelihood_tree_hmms) {
      backend_workspace.ReserveCategorical(
          PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(model.plan, 4,
                                                   storage.batch_capacity, 16,
                                                   model.observation_nodes));
    }
  }
}

void Workspace::ReserveMarginals(PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS(
    AlignmentModelView model, std::size_t site_batch_capacity)) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMarginals,
      [&](auto &backend_workspace, std::size_t batch) {
        backend_workspace.ReserveCategoricalMarginals(
            PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS(model.plan, 4, batch, 16,
                                                     model.observation_nodes));
      });
}

PreparedTimings Workspace::LastTimings() const { return impl_->timings; }

std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView model,
                                               Workspace &workspace) {
  return LogLikelihoodsPrepared(model, workspace, InputUpdate::kAll);
}

std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView model,
                                               Workspace &workspace,
                                               InputUpdate update) {
  Workspace::Impl &storage = *workspace.impl_;
  if (storage.sites != model.sites) {
    throw std::invalid_argument(
        "prepared " PARALLEL_PHYLOGENETICS_GPU_NAME
        " likelihoods require Reserve for this alignment shape");
  }
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kLikelihood, false);
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmms,
      [](auto &backend_workspace) {
        return backend_workspace.CategoricalInputs();
      },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          auto &backend_workspace, InputUpdate batch_update) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            LogPartitionFunctionPrepared(factors, backend_workspace,
                                         batch_update);
      },
      storage.output, update, storage.timings);
}

AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Workspace &workspace) {
  return MaximumAPosterioriPrepared(model, workspace, InputUpdate::kAll);
}

AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Workspace &workspace,
                                                InputUpdate update) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMaximum);
  return internal::MaximumAPosterioriPrepared(
      model, storage.tree_hmms,
      [](auto &backend_workspace) {
        return backend_workspace.CategoricalInputs();
      },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          auto &backend_workspace, InputUpdate batch_update) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            MaximumAPosterioriPrepared(factors, backend_workspace,
                                       batch_update);
      },
      std::span<Scalar>(storage.output).first(model.sites),
      std::span<std::uint32_t>(storage.state_output)
          .first(model.sites * model.plan.num_nodes()),
      std::span<std::uint32_t>(storage.selected_categories).first(model.sites),
      update, storage.timings);
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace) {
  return PosteriorSamplePrepared(model, uniforms, workspace, InputUpdate::kAll);
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const Scalar> uniforms, Workspace &workspace,
                        InputUpdate update) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kSampling);
  return internal::PosteriorSamplePrepared(
      model, uniforms, storage.tree_hmms, storage.sampling_likelihood_tree_hmms,
      [](auto &backend_workspace) {
        return backend_workspace.CategoricalInputs();
      },
      [](auto &backend_workspace, std::size_t batch) {
        return backend_workspace.Uniforms(batch);
      },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          auto &backend_workspace, InputUpdate batch_update) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            LogPartitionFunctionPrepared(factors, backend_workspace,
                                         batch_update);
      },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          std::span<const Scalar> staged_uniforms, auto &backend_workspace,
          InputUpdate batch_update) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            PosteriorSamplePrepared(factors, staged_uniforms, backend_workspace,
                                    batch_update);
      },
      std::span<Scalar>(storage.category_output)
          .first(storage.rate_categories * model.sites),
      std::span<std::uint32_t>(storage.state_output)
          .first(model.sites * model.plan.num_nodes()),
      std::span<std::uint32_t>(storage.selected_categories).first(model.sites),
      update, storage.timings);
}

AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Workspace &workspace) {
  return PosteriorMarginalsPrepared(model, workspace, InputUpdate::kAll);
}

AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Workspace &workspace,
                                                  InputUpdate update) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMarginals);
  return internal::PosteriorMarginalsPrepared(
      model, storage.tree_hmms,
      [](auto &backend_workspace) {
        return backend_workspace.CategoricalInputs();
      },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          auto &backend_workspace, InputUpdate batch_update) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            PosteriorMarginalsPrepared(factors, backend_workspace,
                                       batch_update);
      },
      std::span<Scalar>(storage.output).first(model.sites),
      std::span<Scalar>(storage.node_output)
          .first(model.sites * model.plan.num_nodes() * 4),
      std::span<Scalar>(storage.edge_output)
          .first(model.sites * model.plan.num_edges() * 16),
      std::span<Scalar>(storage.category_output)
          .first(storage.rate_categories * model.sites),
      update, storage.timings);
}

} // namespace parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE

#undef PARALLEL_PHYLOGENETICS_BACKEND_ARGUMENTS
#undef PARALLEL_PHYLOGENETICS_BACKEND_PARAMETERS
#undef PARALLEL_PHYLOGENETICS_DEVICE_ARGUMENT
#undef PARALLEL_PHYLOGENETICS_DEVICE_PARAMETER
