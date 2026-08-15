#ifndef PARALLEL_PHYLOGENETICS_ALIGNMENT_H_
#define PARALLEL_PHYLOGENETICS_ALIGNMENT_H_

#include <array>
#include <cstddef>
#include <memory>
#include <span>

#include "btrc/plan.h"
#include "parallel_phylogenetics/likelihood.h"
#include "tree_hmm/accelerator.h"

namespace parallel_phylogenetics {

struct AlignmentModelView {
  const btrc::Plan &plan;
  std::size_t sites;
  std::span<const Scalar> branch_lengths;
  // Strictly increasing node indices. Phylogenetic alignments ordinarily list
  // the tips, but internal observations are also supported.
  std::span<const btrc::Index> observation_nodes;
  // [site, observation_nodes index].
  std::span<const Nucleotide> observations;
  NucleotideModel nucleotide_model{};
  RateMixtureView rate_mixture{};
};

// Returns a view of a contiguous range of alignment sites without copying its
// observations. This is useful for capacity-bounded accelerator recovery,
// whose outputs are naturally consumed one site batch at a time.
AlignmentModelView SelectSites(AlignmentModelView model, std::size_t first_site,
                               std::size_t site_count);

// Selects one finite-mixture category and folds its rate multiplier into the
// nucleotide model. The returned view has an implicit single rate category.
AlignmentModelView SelectRateCategory(AlignmentModelView model,
                                      std::size_t category);

// One maximum-posterior assignment per site. States have shape [site, node]
// and use the order A=0, C=1, G=2, T=3.
struct AlignmentMaximumView {
  std::span<const Scalar> log_weights;
  std::span<const std::uint32_t> states;
  // Globally shared rate category selected for each site's joint MAP state.
  std::span<const std::uint32_t> rate_categories;
};

// One posterior draw per site. States have shape [site, node] and use the
// order A=0, C=1, G=2, T=3.
struct AlignmentPosteriorSampleView {
  std::span<const std::uint32_t> states;
  std::span<const std::uint32_t> rate_categories;
};

// Posterior probabilities for every site in the supplied alignment view.
// Ancestral states have shape [site, node, A/C/G/T], and substitutions have
// shape [site, edge, parent nucleotide, child nucleotide].
struct AlignmentPosteriorView {
  std::span<const Scalar> log_likelihoods;
  std::span<const Scalar> ancestral_states;
  std::span<const Scalar> substitutions;
  // [site, rate category].
  std::span<const Scalar> rate_categories;
};

// Phylogenetic prepared calls use the generic tree-HMM categorical-input
// lifecycle without redefining it at the application layer.
using InputUpdate = tree_hmm::CategoricalInputUpdate;

// Aggregate timing for one phylogenetic prepared evaluation. Backend fields
// are summed over site batches; evaluation_wall_ms additionally includes
// application-level factor preparation and batching.
struct PreparedTimings {
  tree_hmm::AcceleratorTimings backend;
  double evaluation_wall_ms = 0.0;
  std::size_t site_batches = 0;
};

class AlignmentWorkspace {
public:
  struct Impl;

  AlignmentWorkspace();
  ~AlignmentWorkspace();
  AlignmentWorkspace(AlignmentWorkspace &&) noexcept;
  AlignmentWorkspace &operator=(AlignmentWorkspace &&) noexcept;
  AlignmentWorkspace(const AlignmentWorkspace &) = delete;
  AlignmentWorkspace &operator=(const AlignmentWorkspace &) = delete;

  // Allocates all host factor storage used by subsequent Prepare calls.
  void Reserve(const btrc::Plan &plan, std::size_t sites);

private:
  friend tree_hmm::BatchedModelView Prepare(AlignmentModelView,
                                            AlignmentWorkspace &);
  std::unique_ptr<Impl> impl_;
};

// Converts phylogenetic observations and reversible nucleotide branch models
// into the generic
// tree-HMM batch layout without allocating. The returned spans are owned by
// workspace and remain valid until it is prepared or reserved again.
tree_hmm::BatchedModelView Prepare(AlignmentModelView model,
                                   AlignmentWorkspace &workspace);

// Writes the same factors directly into caller-provided accelerator staging
// storage. This shares the implementation above and performs no allocation.
tree_hmm::BatchedModelView
Prepare(AlignmentModelView model,
        tree_hmm::MutableBatchedModelView destination);

// Prepares the same model using compact categorical observations. Category
// values are the four-bit IUPAC nucleotide masks and the shared emission table
// maps each mask to its compatible hidden states.
tree_hmm::BatchedCategoricalModelView
Prepare(AlignmentModelView model,
        tree_hmm::MutableBatchedCategoricalModelView destination);

class SequentialWorkspace {
public:
  struct Impl;

  SequentialWorkspace();
  ~SequentialWorkspace();
  SequentialWorkspace(SequentialWorkspace &&) noexcept;
  SequentialWorkspace &operator=(SequentialWorkspace &&) noexcept;
  SequentialWorkspace(const SequentialWorkspace &) = delete;
  SequentialWorkspace &operator=(const SequentialWorkspace &) = delete;

  void Reserve(const btrc::Plan &plan, std::size_t sites);

private:
  friend std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView,
                                                        SequentialWorkspace &);
  std::unique_ptr<Impl> impl_;
};

// Conventional postorder Felsenstein pruning with per-node scaling. This is
// the allocation-free sequential CPU baseline for accelerator comparisons.
std::span<const Scalar> LogLikelihoodsPrepared(AlignmentModelView model,
                                               SequentialWorkspace &workspace);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_ALIGNMENT_H_
