#ifndef PARALLEL_PHYLOGENETICS_IO_H_
#define PARALLEL_PHYLOGENETICS_IO_H_

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "btrc/plan.h"
#include "parallel_phylogenetics/likelihood.h"

namespace parallel_phylogenetics {

struct Phylogeny {
  btrc::Plan plan;
  // Plan edge order.
  std::vector<Scalar> branch_lengths;
  // Plan node order. Internal and root labels may be empty.
  std::vector<std::string> labels;
};

struct SequenceRecord {
  std::string name;
  std::string sequence;
};

struct SequenceAlignment {
  std::size_t sites = 0;
  std::vector<SequenceRecord> records;
};

struct EncodedAlignment {
  std::size_t sites = 0;
  // Strictly increasing observed node indices and [site, observed node]
  // values, ready for AlignmentModelView.
  std::vector<btrc::Index> observation_nodes;
  std::vector<Nucleotide> observations;
};

Phylogeny ParseNewick(std::string_view text);
Phylogeny LoadNewick(const std::filesystem::path &path);

SequenceAlignment ParseFasta(std::string_view text);
SequenceAlignment LoadFasta(const std::filesystem::path &path);

// Parses relaxed sequential or interleaved PHYLIP. Continuation blocks may
// omit taxon names or repeat the names in the first block's order.
SequenceAlignment ParsePhylip(std::string_view text);
SequenceAlignment LoadPhylip(const std::filesystem::path &path);

// Matches FASTA record names to leaf labels. Standard IUPAC ambiguity codes,
// gaps, and '?' are represented as unknown observations.
EncodedAlignment EncodeAlignment(const Phylogeny &phylogeny,
                                 const SequenceAlignment &alignment);

} // namespace parallel_phylogenetics

#endif // PARALLEL_PHYLOGENETICS_IO_H_
