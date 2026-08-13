#ifndef PARALLEL_PHYLOGENETICS_ALIGNMENT_INTERNAL_H_
#define PARALLEL_PHYLOGENETICS_ALIGNMENT_INTERNAL_H_

#include "parallel_phylogenetics/alignment.h"

namespace parallel_phylogenetics::internal {

// Prepares the factors shared by every site batch in one alignment
// evaluation. The destination may have less site capacity than model.sites.
void PrepareCategoricalShared(
    AlignmentModelView model,
    tree_hmm::MutableBatchedCategoricalModelView destination);

// Prepares only the observations for one already-validated site batch. The
// shared factors must have been prepared in the same destination first.
tree_hmm::BatchedCategoricalModelView PrepareCategoricalObservations(
    AlignmentModelView model,
    tree_hmm::MutableBatchedCategoricalModelView destination);

}  // namespace parallel_phylogenetics::internal

#endif  // PARALLEL_PHYLOGENETICS_ALIGNMENT_INTERNAL_H_
