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
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kLikelihood, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategorical(model.plan, 4, batch, 16,
                                            model.observation_nodes);
      });
}

void Workspace::ReserveMaximum(AlignmentModelView model,
                               std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMaximum, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalMaximum(model.plan, 4, batch, 16,
                                                   model.observation_nodes);
      });
}

void Workspace::ReserveSampling(AlignmentModelView model,
                                std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kSampling, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalSampling(model.plan, 4, batch, 16,
                                                    model.observation_nodes);
      });
}

void Workspace::ReserveMarginals(AlignmentModelView model,
                                 std::size_t site_batch_capacity) {
  Impl &storage = *impl_;
  internal::ReserveOperation(
      storage, model, site_batch_capacity,
      internal::PreparedOperation::kMarginals, [&](std::size_t batch) {
        storage.tree_hmm.ReserveCategoricalMarginals(model.plan, 4, batch, 16,
                                                     model.observation_nodes);
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
        "prepared Metal likelihoods require Reserve for this alignment shape");
  }
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kLikelihood, false);
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors,
          InputUpdate batch_update) {
        return tree_hmm::metal::LogPartitionFunctionPrepared(factors,
                                                              storage.tree_hmm,
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
      model, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors,
          InputUpdate batch_update) {
        return tree_hmm::metal::MaximumAPosterioriPrepared(factors,
                                                            storage.tree_hmm,
                                                            batch_update);
      },
      update, storage.timings);
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace) {
  return PosteriorSamplePrepared(model, uniforms, workspace,
                                 InputUpdate::kAll);
}

AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace, InputUpdate update) {
  Workspace::Impl &storage = *workspace.impl_;
  internal::ValidatePrepared(model, storage,
                             internal::PreparedOperation::kSampling);
  return internal::PosteriorSamplePrepared(
      model, uniforms, storage.tree_hmm.CategoricalInputs(),
      [&](std::size_t batch) { return storage.tree_hmm.Uniforms(batch); },
      [&](tree_hmm::BatchedCategoricalModelView factors,
          std::span<const Scalar> staged_uniforms, InputUpdate batch_update) {
        return tree_hmm::metal::PosteriorSamplePrepared(
            factors, staged_uniforms, storage.tree_hmm,
            batch_update);
      },
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
      model, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors,
          InputUpdate batch_update) {
        return tree_hmm::metal::PosteriorMarginalsPrepared(factors,
                                                            storage.tree_hmm,
                                                            batch_update);
      },
      update, storage.timings);
}

} // namespace parallel_phylogenetics::metal
