#include "parallel_phylogenetics/metal.h"

#include <stdexcept>
#include <utility>

#include "src/accelerator_internal.h"
#include "tree_hmm/metal.h"

namespace parallel_phylogenetics::metal {

struct Workspace::Impl
    : internal::WorkspaceStorage<tree_hmm::metal::Workspace> {};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() { return tree_hmm::metal::Available(); }

std::string DeviceDescription() { return tree_hmm::metal::DeviceDescription(); }

void Workspace::Reserve(AlignmentModelView model,
                        std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(storage, model, site_batch_capacity,
                             internal::PreparedOperation::kLikelihood,
                             [&](std::size_t batch) {
                               storage.tree_hmm.Reserve(model.plan, 4, batch);
                             });
}

void Workspace::ReserveMaximum(AlignmentModelView model,
                               std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMaximum, [&](std::size_t batch) {
        storage.tree_hmm.ReserveMaximum(model.plan, 4, batch);
      });
}

void Workspace::ReserveSampling(AlignmentModelView model,
                                std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kSampling, [&](std::size_t batch) {
        storage.tree_hmm.ReserveSampling(model.plan, 4, batch);
      });
}

void Workspace::ReserveMarginals(AlignmentModelView model,
                                 std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMarginals, [&](std::size_t batch) {
        storage.tree_hmm.ReserveMarginals(model.plan, 4, batch);
      });
}

std::span<const float> LogLikelihoodsPrepared(AlignmentModelView model,
                                              Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  if (storage.sites != model.sites) {
    throw std::invalid_argument(
        "prepared Metal likelihoods require Reserve for this alignment shape");
  }
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kLikelihood, false);
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmm.Inputs(),
      [&](tree_hmm::BatchedModelView factors) {
        return tree_hmm::metal::LogPartitionFunctionPrepared(factors,
                                                             storage.tree_hmm);
      },
      storage.output);
}

AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMaximum);
  return internal::MaximumAPosterioriPrepared(
      model, storage.tree_hmm.Inputs(),
      [&](tree_hmm::BatchedModelView factors) {
        return tree_hmm::metal::MaximumAPosterioriPrepared(factors,
                                                           storage.tree_hmm);
      });
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const float> uniforms, Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kSampling);
  return internal::PosteriorSamplePrepared(
      model, uniforms, storage.tree_hmm.Inputs(),
      [&](std::size_t batch) { return storage.tree_hmm.Uniforms(batch); },
      [&](tree_hmm::BatchedModelView factors,
          std::span<const float> staged_uniforms) {
        return tree_hmm::metal::PosteriorSamplePrepared(
            factors, staged_uniforms, storage.tree_hmm);
      });
}

AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kMarginals);
  return internal::PosteriorMarginalsPrepared(
      model, storage.tree_hmm.Inputs(),
      [&](tree_hmm::BatchedModelView factors) {
        return tree_hmm::metal::PosteriorMarginalsPrepared(factors,
                                                           storage.tree_hmm);
      });
}

} // namespace parallel_phylogenetics::metal
