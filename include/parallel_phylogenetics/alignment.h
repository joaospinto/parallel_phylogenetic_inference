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
  std::span<const double> branch_lengths;
  // Strictly increasing node indices. Phylogenetic alignments ordinarily list
  // the tips, but internal observations are also supported.
  std::span<const btrc::Index> observation_nodes;
  // [site, observation_nodes index].
  std::span<const Nucleotide> observations;
  std::array<double, 4> root_frequencies{0.25, 0.25, 0.25, 0.25};
  double substitution_rate = 1.0;
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

// Converts phylogenetic observations and JC69 branch models into the generic
// tree-HMM batch layout without allocating. The returned spans are owned by
// workspace and remain valid until it is prepared or reserved again.
tree_hmm::BatchedModelView Prepare(AlignmentModelView model,
                                   AlignmentWorkspace &workspace);

// Writes the same factors directly into caller-provided accelerator staging
// storage. This shares the implementation above and performs no allocation.
tree_hmm::BatchedModelView
Prepare(AlignmentModelView model,
        tree_hmm::MutableBatchedModelView destination);

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
  friend std::span<const double> LogLikelihoodsPrepared(AlignmentModelView,
                                                        SequentialWorkspace &);
  std::unique_ptr<Impl> impl_;
};

// Conventional postorder Felsenstein pruning with per-node scaling. This is
// the allocation-free sequential CPU baseline for accelerator comparisons.
std::span<const double> LogLikelihoodsPrepared(AlignmentModelView model,
                                               SequentialWorkspace &workspace);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_ALIGNMENT_H_
