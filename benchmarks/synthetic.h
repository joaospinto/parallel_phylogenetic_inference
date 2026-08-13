#ifndef PARALLEL_PHYLOGENETICS_BENCHMARK_SYNTHETIC_H_
#define PARALLEL_PHYLOGENETICS_BENCHMARK_SYNTHETIC_H_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "btrc/plan.h"
#include "parallel_phylogenetics/likelihood.h"

namespace parallel_phylogenetics::benchmark {

// The generator is deliberately specified here rather than delegated to a
// standard-library random distribution. SplitMix64 and the cumulative sampler
// make a (topology, leaf count, seed) case reproducible across platforms.
class DeterministicRandom {
public:
  explicit DeterministicRandom(std::uint64_t seed) : state_(seed) {}

  std::uint64_t Next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  double Unit() {
    return static_cast<double>(Next() >> 11) * 0x1.0p-53;
  }

  bool Bit() { return (Next() & 1U) != 0; }

private:
  std::uint64_t state_;
};

inline std::uint64_t SyntheticSeed(std::uint64_t base, std::size_t leaves,
                                   std::size_t patterns,
                                   std::size_t replicate,
                                   std::string_view topology) {
  DeterministicRandom random(base ^ static_cast<std::uint64_t>(leaves));
  std::uint64_t result = random.Next() ^ static_cast<std::uint64_t>(patterns);
  result ^= DeterministicRandom(result + replicate).Next();
  for (const unsigned char character : topology)
    result = DeterministicRandom(result ^ character).Next();
  return result;
}

inline std::uint64_t SyntheticTopologySeed(std::uint64_t base,
                                           std::size_t leaves,
                                           std::size_t replicate,
                                           std::string_view topology) {
  DeterministicRandom random(base ^ static_cast<std::uint64_t>(leaves));
  std::uint64_t result = random.Next();
  result ^= DeterministicRandom(result + replicate).Next();
  for (const unsigned char character : topology)
    result = DeterministicRandom(result ^ character).Next();
  return result;
}

inline std::uint64_t SyntheticSequenceSeed(std::uint64_t topology_seed,
                                           std::size_t raw_sites,
                                           double evolutionary_height) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t result =
      DeterministicRandom(topology_seed ^ static_cast<std::uint64_t>(raw_sites))
          .Next();
  return DeterministicRandom(
             result ^ std::bit_cast<std::uint64_t>(evolutionary_height))
      .Next();
}

struct SyntheticTopology {
  std::vector<std::int64_t> parents;
  std::vector<btrc::Index> leaves;
};

inline std::size_t SampleSplit(std::string_view topology, std::size_t leaves,
                               DeterministicRandom &random,
                               std::span<const double> harmonic_numbers) {
  if (leaves < 2)
    throw std::invalid_argument("a synthetic split requires at least two leaves");
  if (topology == "balanced")
    return leaves / 2;
  if (topology == "caterpillar")
    return random.Bit() ? 1 : leaves - 1;
  if (topology == "yule" || topology == "uniform")
    throw std::logic_error("growth topology reached the split sampler");
  if (topology == "beta-critical") {
    // Aldous's critical beta-splitting distribution (beta = -1):
    // q(n, i) is proportional to 1 / (i (n-i)). The partial fraction
    // identity n/[i(n-i)] = 1/i + 1/(n-i) lets us sample it without
    // constructing O(n) split weights at every internal node: draw a
    // harmonic variate on {1,...,n-1}, then reflect it with probability 1/2.
    if (harmonic_numbers.size() < leaves)
      throw std::invalid_argument("harmonic sampler workspace is too small");
    const double target = random.Unit() * harmonic_numbers[leaves - 1];
    const auto position = std::lower_bound(harmonic_numbers.begin() + 1,
                                           harmonic_numbers.begin() + leaves,
                                           target);
    const std::size_t split =
        static_cast<std::size_t>(position - harmonic_numbers.begin());
    return random.Bit() ? split : leaves - split;
  } else {
    throw std::invalid_argument("unsupported synthetic topology distribution");
  }
}

inline SyntheticTopology MakeGrowthTopology(std::string_view topology,
                                             std::size_t leaves,
                                             DeterministicRandom &random) {
  // A Yule tree is obtained by repeatedly choosing a leaf uniformly and
  // replacing it by a cherry. A PDA tree is obtained by adding a new leaf to
  // an edge chosen uniformly from the planted rooted tree, including the stem
  // edge above the root. Both constructions are exchangeable and linear-time.
  SyntheticTopology result;
  result.parents.assign(2 * leaves - 1, -1);
  std::vector<btrc::Index> current_leaves;
  current_leaves.reserve(leaves);
  current_leaves.push_back(1);
  current_leaves.push_back(2);
  result.parents[1] = 0;
  result.parents[2] = 0;
  btrc::Index next = 3;
  for (std::size_t count = 2; count < leaves; ++count) {
    btrc::Index child = 0;
    if (topology == "yule") {
      const std::size_t chosen = static_cast<std::size_t>(
          random.Next() % current_leaves.size());
      child = current_leaves[chosen];
    } else if (topology == "uniform") {
      child = static_cast<btrc::Index>(
          random.Next() % static_cast<std::uint64_t>(2 * count - 1));
    } else {
      throw std::logic_error("unsupported growth topology");
    }
    const btrc::Index parent = next++;
    const btrc::Index leaf = next++;
    const std::int64_t predecessor = result.parents[child];
    result.parents[parent] = predecessor;
    result.parents[child] = parent;
    result.parents[leaf] = parent;
    current_leaves.push_back(leaf);
  }
  std::sort(current_leaves.begin(), current_leaves.end());
  result.leaves = std::move(current_leaves);
  return result;
}

inline void BuildSyntheticTree(std::string_view topology,
                               std::size_t leaf_count,
                               DeterministicRandom &random,
                               std::span<const double> harmonic_numbers,
                               SyntheticTopology &tree) {
  struct PendingSubtree {
    std::size_t leaves;
    std::int64_t parent;
  };
  std::vector<PendingSubtree> pending;
  pending.reserve(2 * leaf_count - 1);
  pending.push_back({leaf_count, -1});
  while (!pending.empty()) {
    const PendingSubtree subtree = pending.back();
    pending.pop_back();
    if (tree.parents.size() > std::numeric_limits<btrc::Index>::max())
      throw std::length_error("synthetic tree exceeds the planner index limit");
    const auto node = static_cast<btrc::Index>(tree.parents.size());
    tree.parents.push_back(subtree.parent);
    if (subtree.leaves == 1) {
      tree.leaves.push_back(node);
      continue;
    }
    const std::size_t left =
        SampleSplit(topology, subtree.leaves, random, harmonic_numbers);
    // LIFO order preserves the recursive generator's left-before-right draw
    // order without consuming the host call stack on deep trees.
    pending.push_back({subtree.leaves - left, node});
    pending.push_back({left, node});
  }
}

inline SyntheticTopology MakeSyntheticTopology(std::string_view topology,
                                                std::size_t leaves,
                                                std::uint64_t seed) {
  if (leaves < 2)
    throw std::invalid_argument("synthetic trees require at least two leaves");
  if (leaves > (std::numeric_limits<btrc::Index>::max() + std::uint64_t{1}) / 2)
    throw std::length_error("synthetic tree exceeds the planner index limit");
  SyntheticTopology result;
  DeterministicRandom random(seed);
  if (topology == "yule" || topology == "uniform") {
    result = MakeGrowthTopology(topology, leaves, random);
  } else {
    result.parents.reserve(2 * leaves - 1);
    result.leaves.reserve(leaves);
    std::vector<double> harmonic_numbers(leaves);
    for (std::size_t index = 1; index < leaves; ++index)
      harmonic_numbers[index] =
          harmonic_numbers[index - 1] + 1.0 / static_cast<double>(index);
    BuildSyntheticTree(topology, leaves, random, harmonic_numbers, result);
  }
  std::sort(result.leaves.begin(), result.leaves.end());
  return result;
}

inline std::vector<Scalar>
MakeClockLikeBranchLengths(const btrc::Plan &plan,
                           double evolutionary_root_to_tip_distance) {
  if (!(evolutionary_root_to_tip_distance > 0.0) ||
      !std::isfinite(evolutionary_root_to_tip_distance)) {
    throw std::invalid_argument(
        "the evolutionary root-to-tip distance must be finite and positive");
  }
  const std::size_t nodes = plan.num_nodes();
  std::vector<std::vector<std::size_t>> child_edges(nodes);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge)
    child_edges[plan.edge_parents()[edge]].push_back(edge);

  std::vector<btrc::Index> preorder{plan.root()};
  preorder.reserve(nodes);
  for (std::size_t index = 0; index < preorder.size(); ++index) {
    for (const std::size_t edge : child_edges[preorder[index]])
      preorder.push_back(plan.edge_children()[edge]);
  }
  if (preorder.size() != nodes)
    throw std::invalid_argument("clock-like branch lengths require a tree");

  // The backward topological height is zero at every tip and one plus the
  // largest child height at an internal node. Mapping it linearly to
  // evolutionary time makes every edge positive and every root-to-tip path
  // telescope to the requested evolutionary distance, independently of the
  // number of edges on that path.
  std::vector<std::size_t> backward_height(nodes);
  for (auto iterator = preorder.rbegin(); iterator != preorder.rend();
       ++iterator) {
    const btrc::Index node = *iterator;
    for (const std::size_t edge : child_edges[node]) {
      backward_height[node] =
          std::max(backward_height[node],
                   backward_height[plan.edge_children()[edge]] + 1);
    }
  }
  const std::size_t root_height = backward_height[plan.root()];
  if (root_height == 0)
    throw std::invalid_argument("clock-like branch lengths require an edge");

  std::vector<Scalar> lengths(plan.num_edges());
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    const std::size_t parent_height = backward_height[plan.edge_parents()[edge]];
    const std::size_t child_height = backward_height[plan.edge_children()[edge]];
    lengths[edge] = static_cast<Scalar>(
        evolutionary_root_to_tip_distance *
        static_cast<double>(parent_height - child_height) /
        static_cast<double>(root_height));
    if (!(lengths[edge] > Scalar{0}))
      throw std::underflow_error("clock-like branch length rounded to zero");
  }
  return lengths;
}

inline std::vector<double>
RootToTipDistances(const btrc::Plan &plan,
                   std::span<const Scalar> branch_lengths,
                   std::span<const btrc::Index> tips) {
  if (branch_lengths.size() != plan.num_edges())
    throw std::invalid_argument("one branch length is required per edge");
  std::vector<std::vector<std::size_t>> child_edges(plan.num_nodes());
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge)
    child_edges[plan.edge_parents()[edge]].push_back(edge);
  std::vector<double> distance(plan.num_nodes());
  std::vector<btrc::Index> preorder{plan.root()};
  for (std::size_t index = 0; index < preorder.size(); ++index) {
    const btrc::Index parent = preorder[index];
    for (const std::size_t edge : child_edges[parent]) {
      const btrc::Index child = plan.edge_children()[edge];
      distance[child] = distance[parent] + branch_lengths[edge];
      preorder.push_back(child);
    }
  }
  std::vector<double> result;
  result.reserve(tips.size());
  for (const btrc::Index tip : tips) {
    if (tip >= plan.num_nodes() || !child_edges[tip].empty())
      throw std::invalid_argument("root-to-tip distances require unique tips");
    result.push_back(distance[tip]);
  }
  return result;
}

inline std::vector<Nucleotide>
SimulateJukesCantorAlignment(const btrc::Plan &plan,
                             std::span<const Scalar> branch_lengths,
                             std::span<const btrc::Index> tips,
                             std::size_t raw_sites, std::uint64_t seed,
                             Scalar substitution_rate = Scalar{1}) {
  if (raw_sites == 0 || tips.empty())
    throw std::invalid_argument("simulation requires sites and observed tips");
  if (raw_sites > std::numeric_limits<std::size_t>::max() / tips.size())
    throw std::length_error("simulated alignment size overflows size_t");
  if (branch_lengths.size() != plan.num_edges())
    throw std::invalid_argument("one branch length is required per edge");
  if (!std::is_sorted(tips.begin(), tips.end()) ||
      std::adjacent_find(tips.begin(), tips.end()) != tips.end()) {
    throw std::invalid_argument(
        "simulated tip indices must be unique and strictly increasing");
  }

  std::vector<std::vector<std::size_t>> child_edges(plan.num_nodes());
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge)
    child_edges[plan.edge_parents()[edge]].push_back(edge);
  for (const btrc::Index tip : tips) {
    if (tip >= plan.num_nodes() || !child_edges[tip].empty())
      throw std::invalid_argument("an observed simulation node is not a tip");
  }
  std::vector<btrc::Index> preorder{plan.root()};
  std::vector<std::size_t> incoming(plan.num_nodes(), plan.num_edges());
  for (std::size_t index = 0; index < preorder.size(); ++index) {
    for (const std::size_t edge : child_edges[preorder[index]]) {
      const btrc::Index child = plan.edge_children()[edge];
      incoming[child] = edge;
      preorder.push_back(child);
    }
  }
  if (preorder.size() != plan.num_nodes())
    throw std::invalid_argument("JC69 simulation requires a tree");

  constexpr std::array<Nucleotide, 4> kStates{
      Nucleotide::kA, Nucleotide::kC, Nucleotide::kG, Nucleotide::kT};
  std::vector<std::array<Scalar, 16>> transitions(plan.num_edges());
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    transitions[edge] =
        JukesCantorTransition(branch_lengths[edge], substitution_rate);
  }

  DeterministicRandom random(seed);
  std::vector<std::size_t> states(plan.num_nodes());
  std::vector<Nucleotide> observations(raw_sites * tips.size());
  for (std::size_t site = 0; site < raw_sites; ++site) {
    states[plan.root()] = static_cast<std::size_t>(random.Next() & 3U);
    for (std::size_t index = 1; index < preorder.size(); ++index) {
      const btrc::Index child = preorder[index];
      const btrc::Index parent = plan.edge_parents()[incoming[child]];
      const std::size_t parent_state = states[parent];
      const auto &transition = transitions[incoming[child]];
      const double draw = random.Unit();
      double cumulative = 0.0;
      std::size_t child_state = 3;
      for (std::size_t state = 0; state < 3; ++state) {
        cumulative += transition[parent_state * 4 + state];
        if (draw < cumulative) {
          child_state = state;
          break;
        }
      }
      states[child] = child_state;
    }
    for (std::size_t index = 0; index < tips.size(); ++index)
      observations[site * tips.size() + index] = kStates[states[tips[index]]];
  }
  return observations;
}

inline std::vector<Nucleotide>
MakeUniquePatterns(std::size_t patterns, std::size_t leaves,
                   std::uint64_t seed) {
  if (patterns == 0 || leaves == 0)
    throw std::invalid_argument("synthetic pattern and leaf counts must be positive");
  std::size_t representable = 1;
  std::size_t identifying_leaves = 0;
  for (; identifying_leaves < leaves && representable < patterns;
       ++identifying_leaves) {
    if (representable > std::numeric_limits<std::size_t>::max() / 4) {
      representable = std::numeric_limits<std::size_t>::max();
      break;
    }
    representable *= 4;
  }
  if (patterns > representable)
    throw std::invalid_argument("requested more distinct patterns than states permit");

  constexpr std::array<Nucleotide, 4> kStates{
      Nucleotide::kA, Nucleotide::kC, Nucleotide::kG, Nucleotide::kT};
  std::vector<Nucleotide> result(patterns * leaves);
  for (std::size_t pattern = 0; pattern < patterns; ++pattern) {
    DeterministicRandom random(seed ^ (pattern * 0x9e3779b97f4a7c15ULL));
    std::size_t code = pattern;
    for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
      std::size_t state = static_cast<std::size_t>(random.Next() & 3U);
      if (leaf < identifying_leaves) {
        state = code & 3U;
        code >>= 2;
      }
      result[pattern * leaves + leaf] = kStates[state];
    }
  }
  return result;
}

struct TreeShapeStatistics {
  bool binary = true;
  std::size_t height = 0;
  std::uint64_t sackin = 0;
  std::uint64_t colless = 0;
  double normalized_colless = 0.0;
};

inline TreeShapeStatistics ShapeStatistics(const btrc::Plan &plan) {
  const std::size_t nodes = plan.num_nodes();
  std::vector<std::vector<btrc::Index>> children(nodes);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge)
    children[plan.edge_parents()[edge]].push_back(plan.edge_children()[edge]);

  std::vector<std::size_t> depth(nodes);
  std::vector<btrc::Index> preorder{plan.root()};
  for (std::size_t index = 0; index < preorder.size(); ++index) {
    const btrc::Index node = preorder[index];
    for (const btrc::Index child : children[node]) {
      depth[child] = depth[node] + 1;
      preorder.push_back(child);
    }
  }
  if (preorder.size() != nodes)
    throw std::invalid_argument("tree-shape statistics require a connected tree");

  TreeShapeStatistics result;
  std::vector<std::size_t> descendant_leaves(nodes);
  for (auto iterator = preorder.rbegin(); iterator != preorder.rend(); ++iterator) {
    const btrc::Index node = *iterator;
    if (children[node].empty()) {
      descendant_leaves[node] = 1;
      result.height = std::max(result.height, depth[node]);
      result.sackin += depth[node];
      continue;
    }
    for (const btrc::Index child : children[node])
      descendant_leaves[node] += descendant_leaves[child];
    if (children[node].size() == 2) {
      const std::size_t left = descendant_leaves[children[node][0]];
      const std::size_t right = descendant_leaves[children[node][1]];
      result.colless += left > right ? left - right : right - left;
    } else {
      result.binary = false;
    }
  }
  const std::uint64_t leaves = descendant_leaves[plan.root()];
  if (result.binary && leaves > 2) {
    const double maximum =
        static_cast<double>((leaves - 1) * (leaves - 2)) / 2.0;
    result.normalized_colless = static_cast<double>(result.colless) / maximum;
  }
  return result;
}

} // namespace parallel_phylogenetics::benchmark

#endif // PARALLEL_PHYLOGENETICS_BENCHMARK_SYNTHETIC_H_
