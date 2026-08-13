#include "benchmarks/benchmark.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckImpl(bool condition, int line) {
  if (!condition)
    throw std::runtime_error("synthetic benchmark test failed at line " +
                             std::to_string(line));
}

#define Check(condition) CheckImpl((condition), __LINE__)

} // namespace

int main() {
  using namespace parallel_phylogenetics::benchmark;
  constexpr std::size_t kLeaves = 64;
  for (const std::string topology :
       {"balanced", "yule", "beta-critical", "uniform", "caterpillar"}) {
    const SyntheticTopology first = MakeSyntheticTopology(topology, kLeaves, 7);
    const SyntheticTopology second =
        MakeSyntheticTopology(topology, kLeaves, 7);
    Check(first.parents == second.parents);
    Check(first.leaves == second.leaves);
    Check(std::is_sorted(first.leaves.begin(), first.leaves.end()));
    Check(std::adjacent_find(first.leaves.begin(), first.leaves.end()) ==
          first.leaves.end());
    Check(first.parents.size() == 2 * kLeaves - 1);
    Check(first.leaves.size() == kLeaves);
    std::vector<std::size_t> child_counts(first.parents.size());
    for (const std::int64_t parent : first.parents) {
      if (parent >= 0)
        ++child_counts[static_cast<std::size_t>(parent)];
    }
    std::vector<btrc::Index> actual_leaves;
    for (std::size_t node = 0; node < child_counts.size(); ++node) {
      if (child_counts[node] == 0)
        actual_leaves.push_back(static_cast<btrc::Index>(node));
    }
    Check(first.leaves == actual_leaves);
    const btrc::Plan plan = btrc::MakePlan(first.parents);
    const TreeShapeStatistics shape = ShapeStatistics(plan);
    Check(shape.binary);
    Check(shape.height > 0);
    Check(shape.sackin >= kLeaves);
  }

  const SyntheticTopology balanced =
      MakeSyntheticTopology("balanced", kLeaves, 9);
  const SyntheticTopology caterpillar =
      MakeSyntheticTopology("caterpillar", kLeaves, 9);
  const TreeShapeStatistics balanced_shape =
      ShapeStatistics(btrc::MakePlan(balanced.parents));
  const TreeShapeStatistics caterpillar_shape =
      ShapeStatistics(btrc::MakePlan(caterpillar.parents));
  Check(balanced_shape.height == 6);
  Check(balanced_shape.colless == 0);
  Check(caterpillar_shape.height == kLeaves - 1);
  Check(caterpillar_shape.normalized_colless == 1.0);

  // Deep synthetic families use an explicit construction stack rather than
  // the process call stack.
  constexpr std::size_t kLargeLeaves = 32768;
  for (const std::string topology : {"beta-critical", "caterpillar"}) {
    const SyntheticTopology large =
        MakeSyntheticTopology(topology, kLargeLeaves, 31);
    Check(large.parents.size() == 2 * kLargeLeaves - 1);
    Check(large.leaves.size() == kLargeLeaves);
  }

  const auto patterns = MakeUniquePatterns(257, 8, 123);
  Check(patterns.size() == 257 * 8);
  for (std::size_t left = 0; left < 257; ++left) {
    for (std::size_t right = left + 1; right < 257; ++right) {
      Check(!std::equal(patterns.begin() + left * 8,
                        patterns.begin() + (left + 1) * 8,
                        patterns.begin() + right * 8));
    }
  }
  Check(SyntheticSeed(1, 64, 256, 0, "yule") ==
        SyntheticSeed(1, 64, 256, 0, "yule"));
  Check(SyntheticSeed(1, 64, 256, 0, "yule") !=
        SyntheticSeed(1, 64, 256, 1, "yule"));

  for (const std::string topology :
       {"balanced", "yule", "beta-critical", "uniform", "caterpillar"}) {
    Options options;
    options.topology = topology;
    options.leaves = 64;
    options.sites = 8;
    const Problem problem = MakeProblem(options);
    Check(std::is_sorted(problem.observation_nodes.begin(),
                         problem.observation_nodes.end()));
    Check(std::adjacent_find(problem.observation_nodes.begin(),
                             problem.observation_nodes.end()) ==
          problem.observation_nodes.end());
  }

  Problem repeated;
  repeated.observation_nodes = {0, 1};
  repeated.sites = 4;
  repeated.raw_sites = 4;
  repeated.observations = {
      parallel_phylogenetics::Nucleotide::kA,
      parallel_phylogenetics::Nucleotide::kC,
      parallel_phylogenetics::Nucleotide::kG,
      parallel_phylogenetics::Nucleotide::kT,
      parallel_phylogenetics::Nucleotide::kA,
      parallel_phylogenetics::Nucleotide::kC,
      parallel_phylogenetics::Nucleotide::kG,
      parallel_phylogenetics::Nucleotide::kT,
  };
  CompressPatterns(repeated);
  Check(repeated.sites == 2);
  Check((repeated.pattern_weights == std::vector<std::uint64_t>{2, 2}));
  Check(repeated.observations.size() == 4);
}
