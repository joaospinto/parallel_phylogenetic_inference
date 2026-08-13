#include "benchmarks/benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
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
    std::vector<std::size_t> out_degree(plan.num_nodes());
    for (const btrc::Index parent : plan.edge_parents())
      ++out_degree[parent];
    std::vector<btrc::Index> actual_tips;
    for (std::size_t node = 0; node < out_degree.size(); ++node) {
      if (out_degree[node] == 0)
        actual_tips.push_back(static_cast<btrc::Index>(node));
    }
    Check(first.leaves == actual_tips);
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

  for (const std::string topology : {"yule", "beta-critical", "uniform",
                                     "caterpillar"}) {
    const SyntheticTopology clock_topology =
        MakeSyntheticTopology(topology, 32, 71);
    const btrc::Plan clock_plan = btrc::MakePlan(clock_topology.parents);
    constexpr double kEvolutionaryHeight = 0.1;
    const std::vector<parallel_phylogenetics::Scalar> branch_lengths =
        MakeClockLikeBranchLengths(clock_plan, kEvolutionaryHeight);
    for (const double distance : RootToTipDistances(
             clock_plan, branch_lengths, clock_topology.leaves)) {
      Check(std::abs(distance - kEvolutionaryHeight) <= 2e-7);
    }
    const auto first_simulation = SimulateJukesCantorAlignment(
        clock_plan, branch_lengths, clock_topology.leaves, 64, 8128);
    const auto repeated_simulation = SimulateJukesCantorAlignment(
        clock_plan, branch_lengths, clock_topology.leaves, 64, 8128);
    const auto different_simulation = SimulateJukesCantorAlignment(
        clock_plan, branch_lengths, clock_topology.leaves, 64, 8129);
    Check(first_simulation == repeated_simulation);
    Check(first_simulation != different_simulation);
  }

  Options raw_options;
  raw_options.topology = "yule";
  raw_options.leaves = 16;
  raw_options.sites = 256;
  raw_options.seed = 9102;
  raw_options.synthetic_sequence_model = "jc69";
  raw_options.evolutionary_root_to_tip_distance = 0.0001;
  const Problem raw = MakeProblem(raw_options);
  Options compressed_options = raw_options;
  compressed_options.compress_patterns = true;
  const Problem compressed = MakeProblem(compressed_options);
  Check(raw.dataset == "synthetic-jc69");
  Check(raw.raw_sites == raw_options.sites);
  Check(raw.sites == raw_options.sites);
  Check(compressed.raw_sites == raw_options.sites);
  Check(compressed.sites < compressed.raw_sites);
  Check(std::accumulate(compressed.pattern_weights.begin(),
                        compressed.pattern_weights.end(), std::uint64_t{0}) ==
        compressed.raw_sites);
  Check(raw.branch_lengths == compressed.branch_lengths);
  Check(raw.observation_nodes == compressed.observation_nodes);

  Options longer_options = raw_options;
  longer_options.sites = 512;
  const Problem longer = MakeProblem(longer_options);
  Check(std::equal(raw.plan.edge_parents().begin(),
                   raw.plan.edge_parents().end(),
                   longer.plan.edge_parents().begin()));
  Check(std::equal(raw.plan.edge_children().begin(),
                   raw.plan.edge_children().end(),
                   longer.plan.edge_children().begin()));
  Check(raw.branch_lengths == longer.branch_lengths);
  Check(raw.seed != longer.seed);
  Options taller_options = raw_options;
  taller_options.evolutionary_root_to_tip_distance = 0.01;
  const Problem taller = MakeProblem(taller_options);
  Check(std::equal(raw.plan.edge_parents().begin(),
                   raw.plan.edge_parents().end(),
                   taller.plan.edge_parents().begin()));
  Check(std::equal(raw.plan.edge_children().begin(),
                   raw.plan.edge_children().end(),
                   taller.plan.edge_children().begin()));
  Check(raw.branch_lengths != taller.branch_lengths);
  Check(raw.seed != taller.seed);

  parallel_phylogenetics::SequentialWorkspace raw_workspace;
  raw_workspace.Reserve(raw.plan, raw.sites);
  const auto raw_values = parallel_phylogenetics::LogLikelihoodsPrepared(
      {raw.plan, raw.sites, raw.branch_lengths, raw.observation_nodes,
       raw.observations},
      raw_workspace);
  parallel_phylogenetics::SequentialWorkspace compressed_workspace;
  compressed_workspace.Reserve(compressed.plan, compressed.sites);
  const auto compressed_values =
      parallel_phylogenetics::LogLikelihoodsPrepared(
          {compressed.plan, compressed.sites, compressed.branch_lengths,
           compressed.observation_nodes, compressed.observations},
          compressed_workspace);
  double raw_log_likelihood = 0.0;
  for (const auto value : raw_values)
    raw_log_likelihood += value;
  double compressed_log_likelihood = 0.0;
  for (std::size_t pattern = 0; pattern < compressed.sites; ++pattern) {
    compressed_log_likelihood += compressed.pattern_weights[pattern] *
                                 compressed_values[pattern];
  }
  Check(std::abs(raw_log_likelihood - compressed_log_likelihood) <=
        2e-6 * std::max(1.0, std::abs(raw_log_likelihood)));
}
