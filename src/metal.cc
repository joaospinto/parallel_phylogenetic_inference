#include "parallel_phylogenetics/metal.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/accelerator_internal.h"
#include "tree_hmm/metal.h"

namespace parallel_phylogenetics::metal {

struct Workspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t sites = 0;
  std::size_t batch_capacity = 0;
  std::vector<btrc::Index> observation_nodes;
  std::vector<float> output;
  tree_hmm::metal::Workspace tree_hmm;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() { return tree_hmm::metal::Available(); }

std::string DeviceDescription() { return tree_hmm::metal::DeviceDescription(); }

void Workspace::Reserve(AlignmentModelView model,
                        std::size_t site_batch_capacity) {
  if (model.sites == 0)
    throw std::invalid_argument("an alignment must contain at least one site");
  const std::size_t batch =
      site_batch_capacity == 0 ? model.sites : site_batch_capacity;
  if (batch > model.sites)
    throw std::invalid_argument(
        "site batch capacity cannot exceed the alignment site count");
  Impl &storage = *impl_;
  storage.tree_hmm.Reserve(model.plan, 4, batch);
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
        "prepared Metal likelihoods require Reserve for this alignment shape");
  }
  return internal::LogLikelihoodsPrepared(
      model, storage.batch_capacity, storage.tree_hmm.Inputs(),
      [&](tree_hmm::BatchedModelView factors) {
        return tree_hmm::metal::LogPartitionFunctionPrepared(factors,
                                                             storage.tree_hmm);
      },
      storage.output);
}

} // namespace parallel_phylogenetics::metal
