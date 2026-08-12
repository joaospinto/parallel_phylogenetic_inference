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
  void ReserveMaximum(AlignmentModelView model,
                      std::size_t site_batch_capacity = 0, int device = 0);
  void ReserveSampling(AlignmentModelView model,
                       std::size_t site_batch_capacity = 0, int device = 0);
  void ReserveMarginals(AlignmentModelView model,
                        std::size_t site_batch_capacity = 0, int device = 0);

private:
  friend std::span<const float> LogLikelihoodsPrepared(AlignmentModelView,
                                                       Workspace &);
  friend AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView,
                                                         Workspace &);
  friend AlignmentPosteriorSampleView
  PosteriorSamplePrepared(AlignmentModelView, std::span<const float>,
                          Workspace &);
  friend AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView,
                                                           Workspace &);
  std::unique_ptr<Impl> impl_;
};

// Evaluates every site without resizing numerical workspace storage or
// rebuilding the contraction plan. The returned values are owned by workspace
// and remain valid until its next evaluation, reservation, move, or
// destruction.
std::span<const float> LogLikelihoodsPrepared(AlignmentModelView model,
                                              Workspace &workspace);

// The recovery functions evaluate the supplied alignment view as one batch;
// its site count must not exceed the reserved capacity. Use SelectSites to
// consume a larger alignment in bounded batches. Returned spans are owned by
// workspace and remain valid until its next evaluation or reservation.
AlignmentMaximumView MaximumAPosterioriPrepared(AlignmentModelView model,
                                                Workspace &workspace);
AlignmentPosteriorSampleView
PosteriorSamplePrepared(AlignmentModelView model,
                        std::span<const float> uniforms, Workspace &workspace);
AlignmentPosteriorView PosteriorMarginalsPrepared(AlignmentModelView model,
                                                  Workspace &workspace);

} // namespace parallel_phylogenetics::cuda

#endif // PARALLEL_PHYLOGENETICS_CUDA_H_
