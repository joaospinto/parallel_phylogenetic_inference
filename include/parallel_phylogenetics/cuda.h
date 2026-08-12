#ifndef PARALLEL_PHYLOGENETICS_CUDA_H_
#define PARALLEL_PHYLOGENETICS_CUDA_H_

#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "parallel_phylogenetics/alignment.h"

namespace parallel_phylogenetics::cuda {

bool Available();
std::string DeviceDescription(int device = 0);

class Workspace {
public:
  struct Impl;

  Workspace();
  ~Workspace();
  Workspace(Workspace &&) noexcept;
  Workspace &operator=(Workspace &&) noexcept;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  // Allocates the application-owned numerical storage used by subsequent
  // prepared evaluations. At most site_batch_capacity sites are evaluated at
  // once; zero selects all sites.
  void Reserve(AlignmentModelView model, std::size_t site_batch_capacity = 0,
               int device = 0);

private:
  friend std::span<const float> LogLikelihoodsPrepared(AlignmentModelView,
                                                       Workspace &);
  std::unique_ptr<Impl> impl_;
};

// Evaluates every site without resizing numerical workspace storage or
// rebuilding the contraction plan. The returned values are owned by workspace
// and remain valid until its next evaluation, reservation, move, or
// destruction.
std::span<const float> LogLikelihoodsPrepared(AlignmentModelView model,
                                              Workspace &workspace);

} // namespace parallel_phylogenetics::cuda

#endif // PARALLEL_PHYLOGENETICS_CUDA_H_
