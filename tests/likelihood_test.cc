#include "parallel_phylogenetics/likelihood.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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
      N::kUnknown, N::kUnknown, N::kUnknown, N::kUnknown,
      N::kA,       N::kG,       N::kT,
  };
  const std::array<double, 4> frequencies{0.3, 0.2, 0.2, 0.3};
  const parallel_phylogenetics::SiteModelView model{
      plan, lengths, observations, frequencies, 1.0};
  const double expected =
      FelsensteinReference(plan, lengths, observations, frequencies);
  const double actual = parallel_phylogenetics::SiteLikelihood(model);
  Check(Near(actual, expected));

  const auto posterior =
      parallel_phylogenetics::AncestralPosterior(model);
  Check(Near(posterior.likelihood, expected));
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    double sum = 0.0;
    for (std::size_t state = 0; state < 4; ++state)
      sum += posterior.ancestral_states[node * 4 + state];
    Check(Near(sum, 1.0));
  }
}
