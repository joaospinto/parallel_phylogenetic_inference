#include "parallel_phylogenetics/alignment.h"
#include "parallel_phylogenetics/io.h"
#include "parallel_phylogenetics/likelihood.h"
#include "tree_hmm/inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
bool g_count_allocations = false;
std::size_t g_allocations = 0;
} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations)
    ++g_allocations;
  if (void *result = std::malloc(size))
    return result;
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

bool Near(double left, double right, double tolerance = 1e-11) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

void Check(bool condition) {
  if (!condition)
    throw std::runtime_error("phylogenetic likelihood test failed");
}

double FelsensteinReference(
    const btrc::Plan &plan, const std::vector<double> &branch_lengths,
    const std::vector<parallel_phylogenetics::Nucleotide> &observations,
    const std::array<double, 4> &root_frequencies) {
  std::vector<std::vector<std::size_t>> children(plan.num_nodes());
  std::vector<std::size_t> incoming(plan.num_nodes(), plan.num_edges());
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    children[plan.edge_parents()[edge]].push_back(plan.edge_children()[edge]);
    incoming[plan.edge_children()[edge]] = edge;
  }
  std::vector<std::size_t> order;
  std::vector<std::size_t> stack{plan.root()};
  while (!stack.empty()) {
    const std::size_t node = stack.back();
    stack.pop_back();
    order.push_back(node);
    stack.insert(stack.end(), children[node].begin(), children[node].end());
  }
  std::vector<double> partials(plan.num_nodes() * 4, 1.0);
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    if (observations[node] == parallel_phylogenetics::Nucleotide::kUnknown)
      continue;
    std::fill(partials.begin() + node * 4, partials.begin() + (node + 1) * 4,
              0.0);
    partials[node * 4 + static_cast<int>(observations[node])] = 1.0;
  }
  for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
    const std::size_t node = *iterator;
    for (const std::size_t child : children[node]) {
      const auto transition = parallel_phylogenetics::JukesCantorTransition(
          branch_lengths[incoming[child]]);
      for (std::size_t parent_state = 0; parent_state < 4; ++parent_state) {
        double message = 0.0;
        for (std::size_t child_state = 0; child_state < 4; ++child_state) {
          message += transition[parent_state * 4 + child_state] *
                     partials[child * 4 + child_state];
        }
        partials[node * 4 + parent_state] *= message;
      }
    }
  }
  double likelihood = 0.0;
  for (std::size_t state = 0; state < 4; ++state)
    likelihood += root_frequencies[state] * partials[plan.root() * 4 + state];
  return likelihood;
}

} // namespace

int main() {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 3, 2});
  const std::vector<double> lengths{0.1, 0.3, 0.2, 0.4, 0.15, 0.5};
  using N = parallel_phylogenetics::Nucleotide;
  const std::vector<N> observations{
      N::kUnknown, N::kUnknown, N::kUnknown, N::kUnknown, N::kA, N::kG, N::kT,
  };
  const std::array<double, 4> frequencies{0.3, 0.2, 0.2, 0.3};
  const parallel_phylogenetics::SiteModelView model{plan, lengths, observations,
                                                    frequencies, 1.0};
  const double expected =
      FelsensteinReference(plan, lengths, observations, frequencies);
  const double actual = parallel_phylogenetics::SiteLikelihood(model);
  Check(Near(actual, expected));
  Check(Near(parallel_phylogenetics::SiteLogLikelihood(model),
             std::log(expected)));

  const auto posterior = parallel_phylogenetics::AncestralPosterior(model);
  Check(Near(posterior.likelihood, expected));
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    double sum = 0.0;
    for (std::size_t state = 0; state < 4; ++state)
      sum += posterior.ancestral_states[node * 4 + state];
    Check(Near(sum, 1.0));
  }

  const std::vector<N> alignment_observations{
      N::kUnknown, N::kUnknown, N::kUnknown, N::kUnknown, N::kA, N::kG, N::kT,
      N::kUnknown, N::kUnknown, N::kUnknown, N::kUnknown, N::kC, N::kC, N::kA,
  };
  parallel_phylogenetics::AlignmentWorkspace alignment_workspace;
  alignment_workspace.Reserve(plan, 2);
  const tree_hmm::BatchedModelView prepared = parallel_phylogenetics::Prepare(
      {plan, 2, lengths, alignment_observations, frequencies, 1.0},
      alignment_workspace);
  Check(prepared.batch == 2);
  Check(prepared.states == 4);
  std::vector<float> direct_nodes(prepared.node_potentials.size());
  std::vector<float> direct_edges(prepared.edge_potentials.size());
  tree_hmm::MutableBatchedModelView direct_destination{plan, 4, 2, direct_nodes,
                                                       direct_edges};
  const tree_hmm::BatchedModelView direct = parallel_phylogenetics::Prepare(
      {plan, 2, lengths, alignment_observations, frequencies, 1.0},
      direct_destination);
  Check(std::equal(prepared.node_potentials.begin(),
                   prepared.node_potentials.end(),
                   direct.node_potentials.begin()));
  Check(std::equal(prepared.edge_potentials.begin(),
                   prepared.edge_potentials.end(),
                   direct.edge_potentials.begin()));
  parallel_phylogenetics::SequentialWorkspace sequential_workspace;
  sequential_workspace.Reserve(plan, 2);
  const std::span<const double> sequential =
      parallel_phylogenetics::LogLikelihoodsPrepared(
          {plan, 2, lengths, alignment_observations, frequencies, 1.0},
          sequential_workspace);
  for (std::size_t site = 0; site < 2; ++site) {
    const std::size_t node_values = plan.num_nodes() * 4;
    std::vector<double> site_nodes(
        prepared.node_potentials.begin() + site * node_values,
        prepared.node_potentials.begin() + (site + 1) * node_values);
    std::vector<double> site_edges(prepared.edge_potentials.begin(),
                                   prepared.edge_potentials.end());
    const double prepared_log_likelihood =
        tree_hmm::LogPartitionFunction({plan, 4, site_nodes, site_edges});
    const std::span<const N> site_observations(alignment_observations.data() +
                                                   site * plan.num_nodes(),
                                               plan.num_nodes());
    Check(Near(prepared_log_likelihood,
               parallel_phylogenetics::SiteLogLikelihood(
                   {plan, lengths, site_observations, frequencies, 1.0}),
               2e-6));
    Check(Near(sequential[site], prepared_log_likelihood, 2e-6));
  }
  g_allocations = 0;
  g_count_allocations = true;
  for (int repeat = 0; repeat < 10; ++repeat) {
    static_cast<void>(parallel_phylogenetics::Prepare(
        {plan, 2, lengths, alignment_observations, frequencies, 1.0},
        alignment_workspace));
    static_cast<void>(parallel_phylogenetics::Prepare(
        {plan, 2, lengths, alignment_observations, frequencies, 1.0},
        direct_destination));
    static_cast<void>(parallel_phylogenetics::LogLikelihoodsPrepared(
        {plan, 2, lengths, alignment_observations, frequencies, 1.0},
        sequential_workspace));
  }
  g_count_allocations = false;
  Check(g_allocations == 0);

  const parallel_phylogenetics::Phylogeny parsed_tree =
      parallel_phylogenetics::ParseNewick(
          "((A:0.1,'B_taxon':0.2)I:0.3,C:0.4)Root;");
  Check(parsed_tree.plan.num_nodes() == 5);
  Check(parsed_tree.plan.num_edges() == 4);
  Check(parsed_tree.labels[parsed_tree.plan.root()] == "Root");
  Check(Near(parsed_tree.branch_lengths[0], 0.3));
  Check(Near(parsed_tree.branch_lengths[1], 0.1));
  Check(Near(parsed_tree.branch_lengths[2], 0.2));
  Check(Near(parsed_tree.branch_lengths[3], 0.4));

  const parallel_phylogenetics::SequenceAlignment parsed_alignment =
      parallel_phylogenetics::ParseFasta(
          ">A description\nACG\n>B_taxon description\nA-N\n>C\nTCG\n");
  Check(parsed_alignment.sites == 3);
  const parallel_phylogenetics::EncodedAlignment encoded =
      parallel_phylogenetics::EncodeAlignment(parsed_tree, parsed_alignment);
  Check(encoded.sites == 3);
  Check(encoded.observations.size() ==
        encoded.sites * parsed_tree.plan.num_nodes());
}
