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

std::string DeviceDescription(int device) {
  return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::DeviceDescription(
      device);
}

void Workspace::Reserve(AlignmentModelView model,
                        std::size_t site_batch_capacity, int device) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kLikelihood, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategorical(model.plan, 4, batch, 16,
                                            model.observation_nodes, device);
      });
}

void Workspace::ReserveMaximum(AlignmentModelView model,
                               std::size_t site_batch_capacity, int device) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMaximum, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalMaximum(
            model.plan, 4, batch, 16, model.observation_nodes, device);
      });
}

void Workspace::ReserveSampling(AlignmentModelView model,
                                std::size_t site_batch_capacity, int device) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kSampling, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalSampling(
            model.plan, 4, batch, 16, model.observation_nodes, device);
      });
}

void Workspace::ReserveMarginals(AlignmentModelView model,
                                 std::size_t site_batch_capacity, int device) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMarginals, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalMarginals(
            model.plan, 4, batch, 16, model.observation_nodes, device);
      });
}

std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView model,
                                               Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  if (storage.sites != model.sites) {
    throw std::invalid_argument(
        "prepared " PARALLEL_PHYLOGENETICS_GPU_NAME
        " likelihoods require Reserve for this alignment shape");
  }
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kLikelihood, false);
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            LogPartitionFunctionPrepared(factors, storage.tree_hmm);
      },
      storage.output);
}

AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMaximum);
  return internal::MaximumAPosterioriPrepared(
      model, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            MaximumAPosterioriPrepared(factors, storage.tree_hmm);
      });
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kSampling);
  return internal::PosteriorSamplePrepared(
      model, uniforms, storage.tree_hmm.CategoricalInputs(),
      [&](std::size_t batch) { return storage.tree_hmm.Uniforms(batch); },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          std::span<const Scalar> staged_uniforms) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            PosteriorSamplePrepared(factors, staged_uniforms, storage.tree_hmm);
      });
}

AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMarginals);
  return internal::PosteriorMarginalsPrepared(
      model, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors) {
        return tree_hmm::PARALLEL_PHYLOGENETICS_TREE_HMM_NAMESPACE::
            PosteriorMarginalsPrepared(factors, storage.tree_hmm);
      });
}

} // namespace parallel_phylogenetics::PARALLEL_PHYLOGENETICS_GPU_NAMESPACE
