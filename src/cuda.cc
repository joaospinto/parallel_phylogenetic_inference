#include "parallel_phylogenetics/cuda.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/accelerator_internal.h"
#include "tree_hmm/cuda.h"

namespace parallel_phylogenetics::cuda {

struct Workspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t sites = 0;
  std::size_t batch_capacity = 0;
  std::vector<btrc::Index> observation_nodes;
  std::vector<float> output;
  tree_hmm::cuda::Workspace tree_hmm;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() { return tree_hmm::cuda::Available(); }

std::string DeviceDescription(int device) {
  return tree_hmm::cuda::DeviceDescription(device);
}

void Workspace::Reserve(AlignmentModelView model,
                        std::size_t site_batch_capacity, int device) {
  if (model.sites == 0)
    throw std::invalid_argument("an alignment must contain at least one site");
  const std::size_t batch =
      site_batch_capacity == 0 ? model.sites : site_batch_capacity;
  if (batch > model.sites)
    throw std::invalid_argument(
        "site batch capacity cannot exceed the alignment site count");
  Impl &storage = *impl_;
  storage.tree_hmm.ReserveCategorical(model.plan, 4, batch, 16,
                                      model.observation_nodes, device);
  storage.plan = &model.plan;
  storage.sites = model.sites;
  storage.batch_capacity = batch;
  storage.observation_nodes.assign(model.observation_nodes.begin(),
                                   model.observation_nodes.end());
  storage.output.resize(model.sites);
}

std::span<const float> LogLikelihoodsPrepared(AlignmentModelView model,
                                              Workspace &workspace) {
  Workspace::Impl &storage = *workspace.impl_;
  if (storage.plan != &model.plan || storage.sites != model.sites ||
      storage.observation_nodes.size() != model.observation_nodes.size() ||
      !std::equal(
          storage.observation_nodes.begin(), storage.observation_nodes.end(),
          model.observation_nodes.begin(), model.observation_nodes.end())) {
    throw std::invalid_argument(
        "prepared CUDA likelihoods require Reserve for this alignment shape");
  }
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmm.CategoricalInputs(),
      [&](tree_hmm::BatchedCategoricalModelView factors) {
        return tree_hmm::cuda::LogPartitionFunctionPrepared(factors,
                                                            storage.tree_hmm);
      },
      storage.output);
}

} // namespace parallel_phylogenetics::cuda
