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
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
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

constexpr double kTolerance =
    std::is_same_v<parallel_phylogenetics::Scalar, float> ? 2e-5 : 1e-11;

bool Near(double left, double right, double tolerance = kTolerance) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

void CheckImpl(bool condition, int line) {
  if (!condition)
    throw std::runtime_error("phylogenetic likelihood test failed at line " +
                             std::to_string(line));
}

#define Check(condition) CheckImpl((condition), __LINE__)

double FelsensteinReference(
    const btrc::Plan &plan,
    const std::vector<parallel_phylogenetics::Scalar> &branch_lengths,
    const std::vector<parallel_phylogenetics::Nucleotide> &observations,
    const std::array<parallel_phylogenetics::Scalar, 4> &root_frequencies) {
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
  std::vector<parallel_phylogenetics::Scalar> partials(plan.num_nodes() * 4,
                                                       1.0);
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    for (std::size_t state = 0; state < 4; ++state) {
      if (!parallel_phylogenetics::AllowsState(observations[node], state))
        partials[node * 4 + state] = 0.0;
    }
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
  const auto tiny_transition = parallel_phylogenetics::JukesCantorTransition(
      static_cast<parallel_phylogenetics::Scalar>(1e-12));
  Check(tiny_transition[1] > parallel_phylogenetics::Scalar{0});
  Check(Near(
      tiny_transition[0] +
          parallel_phylogenetics::Scalar{3} * tiny_transition[1],
      1.0,
      8.0 * std::numeric_limits<parallel_phylogenetics::Scalar>::epsilon()));

  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 3, 2});
  const std::vector<parallel_phylogenetics::Scalar> lengths{0.1, 0.3,  0.2,
                                                            0.4, 0.15, 0.5};
  using N = parallel_phylogenetics::Nucleotide;
  const std::vector<N> observations{
      N::kUnknown, N::kUnknown, N::kUnknown, N::kUnknown, N::kA, N::kG, N::kT,
  };
  const std::array<parallel_phylogenetics::Scalar, 4> frequencies{0.3, 0.2, 0.2,
                                                                  0.3};
  parallel_phylogenetics::NucleotideModel nucleotide_model =
      parallel_phylogenetics::JukesCantorModel();
  std::copy(frequencies.begin(), frequencies.end(),
            nucleotide_model.root_frequencies.begin());
  const parallel_phylogenetics::SiteModelView model{plan, lengths, observations,
                                                    nucleotide_model};
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

  const parallel_phylogenetics::NucleotideModel gtr =
      parallel_phylogenetics::GeneralTimeReversibleModel(
          {0.31, 0.19, 0.27, 0.23}, {1.2, 3.1, 0.7, 1.8, 4.6, 0.9});
  const parallel_phylogenetics::RateMixture gamma =
      parallel_phylogenetics::DiscreteGammaRateMixture(0.5, 4);
  const parallel_phylogenetics::SiteModelView mixture_site{
      plan, lengths, observations, gtr, gamma.view()};
  const double mixture_log =
      parallel_phylogenetics::SiteLogLikelihood(mixture_site);
  double explicit_mixture = 0.0;
  for (std::size_t category = 0; category < gamma.rates().size(); ++category) {
    explicit_mixture += gamma.weights()[category] *
                        std::exp(parallel_phylogenetics::SiteLogLikelihood(
                            parallel_phylogenetics::SelectRateCategory(
                                mixture_site, category)));
  }
  Check(Near(mixture_log, std::log(explicit_mixture)));
  const auto mixture_posterior =
      parallel_phylogenetics::AncestralPosterior(mixture_site);
  Check(Near(mixture_posterior.likelihood, explicit_mixture));
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    const double sum = std::accumulate(
        mixture_posterior.ancestral_states.begin() + node * 4,
        mixture_posterior.ancestral_states.begin() + (node + 1) * 4, 0.0);
    Check(Near(sum, 1.0));
  }

  const std::vector<btrc::Index> observation_nodes{4, 5, 6};
  const std::vector<N> alignment_observations{N::kR, N::kG, N::kT,
                                              N::kC, N::kC, N::kA};
  parallel_phylogenetics::AlignmentWorkspace alignment_workspace;
  alignment_workspace.Reserve(plan, 2);
  const tree_hmm::BatchedModelView prepared = parallel_phylogenetics::Prepare(
      {plan, 2, lengths, observation_nodes, alignment_observations,
       nucleotide_model},
      alignment_workspace);
  Check(prepared.batch == 2);
  Check(prepared.states == 4);
  std::vector<parallel_phylogenetics::Scalar> direct_nodes(
      prepared.node_potentials.size());
  std::vector<parallel_phylogenetics::Scalar> direct_edges(
      prepared.edge_potentials.size());
  tree_hmm::MutableBatchedModelView direct_destination{plan, 4, 2, direct_nodes,
                                                       direct_edges};
  const tree_hmm::BatchedModelView direct = parallel_phylogenetics::Prepare(
      {plan, 2, lengths, observation_nodes, alignment_observations,
       nucleotide_model},
      direct_destination);
  Check(std::equal(prepared.node_potentials.begin(),
                   prepared.node_potentials.end(),
                   direct.node_potentials.begin()));
  Check(std::equal(prepared.edge_potentials.begin(),
                   prepared.edge_potentials.end(),
                   direct.edge_potentials.begin()));
  std::vector<std::uint8_t> categorical_observations(
      alignment_observations.size());
  std::vector<parallel_phylogenetics::Scalar> categorical_root(4);
  std::vector<parallel_phylogenetics::Scalar> categorical_emissions(16 * 4);
  std::vector<parallel_phylogenetics::Scalar> categorical_edges(
      prepared.edge_potentials.size());
  tree_hmm::MutableBatchedCategoricalModelView categorical_destination{
      plan,
      4,
      2,
      16,
      observation_nodes,
      categorical_observations,
      categorical_root,
      categorical_emissions,
      categorical_edges};
  const tree_hmm::BatchedCategoricalModelView categorical =
      parallel_phylogenetics::Prepare({plan, 2, lengths, observation_nodes,
                                       alignment_observations,
                                       nucleotide_model},
                                      categorical_destination);
  for (std::size_t site = 0; site < 2; ++site) {
    std::vector<parallel_phylogenetics::Scalar> reconstructed(
        plan.num_nodes() * 4, 1.0f);
    std::copy(categorical.root_potential.begin(),
              categorical.root_potential.end(),
              reconstructed.begin() + plan.root() * 4);
    for (std::size_t observation = 0; observation < observation_nodes.size();
         ++observation) {
      const std::uint8_t category =
          categorical
              .observations[site * observation_nodes.size() + observation];
      for (std::size_t state = 0; state < 4; ++state) {
        reconstructed[observation_nodes[observation] * 4 + state] *=
            categorical.emission_potentials[category * 4 + state];
      }
    }
    Check(std::equal(reconstructed.begin(), reconstructed.end(),
                     prepared.node_potentials.begin() +
                         site * plan.num_nodes() * 4));
  }
  Check(std::equal(categorical.edge_potentials.begin(),
                   categorical.edge_potentials.end(),
                   prepared.edge_potentials.begin()));
  parallel_phylogenetics::SequentialWorkspace sequential_workspace;
  sequential_workspace.Reserve(plan, 2);
  const std::span<const parallel_phylogenetics::Scalar> sequential =
      parallel_phylogenetics::LogLikelihoodsPrepared(
          {plan, 2, lengths, observation_nodes, alignment_observations,
           nucleotide_model},
          sequential_workspace);
  const std::vector<parallel_phylogenetics::Scalar> sequential_values(
      sequential.begin(), sequential.end());
  const parallel_phylogenetics::AlignmentModelView mixture_alignment{
      plan, 2,           lengths, observation_nodes, alignment_observations,
      gtr,  gamma.view()};
  const std::span<const parallel_phylogenetics::Scalar> mixture_sequential =
      parallel_phylogenetics::LogLikelihoodsPrepared(mixture_alignment,
                                                     sequential_workspace);
  for (std::size_t site = 0; site < 2; ++site) {
    const std::size_t node_values = plan.num_nodes() * 4;
    std::vector<parallel_phylogenetics::Scalar> site_nodes(
        prepared.node_potentials.begin() + site * node_values,
        prepared.node_potentials.begin() + (site + 1) * node_values);
    std::vector<parallel_phylogenetics::Scalar> site_edges(
        prepared.edge_potentials.begin(), prepared.edge_potentials.end());
    const double prepared_log_likelihood =
        tree_hmm::LogPartitionFunction({plan, 4, site_nodes, site_edges});
    std::vector<N> site_observations(plan.num_nodes(), N::kUnknown);
    for (std::size_t index = 0; index < observation_nodes.size(); ++index) {
      site_observations[observation_nodes[index]] =
          alignment_observations[site * observation_nodes.size() + index];
    }
    Check(Near(prepared_log_likelihood,
               parallel_phylogenetics::SiteLogLikelihood(
                   {plan, lengths, site_observations, nucleotide_model}),
               2e-6));
    Check(Near(sequential_values[site], prepared_log_likelihood, 2e-6));
    std::vector<N> site_observations_for_mixture(plan.num_nodes(), N::kUnknown);
    for (std::size_t index = 0; index < observation_nodes.size(); ++index) {
      site_observations_for_mixture[observation_nodes[index]] =
          alignment_observations[site * observation_nodes.size() + index];
    }
    Check(Near(
        mixture_sequential[site],
        parallel_phylogenetics::SiteLogLikelihood(
            {plan, lengths, site_observations_for_mixture, gtr, gamma.view()}),
        2e-5));
  }
  g_allocations = 0;
  g_count_allocations = true;
  for (int repeat = 0; repeat < 10; ++repeat) {
    static_cast<void>(parallel_phylogenetics::Prepare(
        {plan, 2, lengths, observation_nodes, alignment_observations,
         nucleotide_model},
        alignment_workspace));
    static_cast<void>(parallel_phylogenetics::Prepare(
        {plan, 2, lengths, observation_nodes, alignment_observations,
         nucleotide_model},
        direct_destination));
    static_cast<void>(parallel_phylogenetics::LogLikelihoodsPrepared(
        {plan, 2, lengths, observation_nodes, alignment_observations,
         nucleotide_model},
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
          ">A description\nRYSW\n>B_taxon description\nKMBD\n>C\nHVN-\n");
  Check(parsed_alignment.sites == 4);
  const parallel_phylogenetics::EncodedAlignment encoded =
      parallel_phylogenetics::EncodeAlignment(parsed_tree, parsed_alignment);
  Check(encoded.sites == 4);
  Check(encoded.observation_nodes.size() == 3);
  Check(encoded.observations.size() ==
        encoded.sites * encoded.observation_nodes.size());
  const std::array<N, 12> expected_ambiguities{
      N::kR, N::kK, N::kH,       N::kY, N::kM, N::kV,
      N::kS, N::kB, N::kUnknown, N::kW, N::kD, N::kUnknown};
  Check(std::equal(encoded.observations.begin(), encoded.observations.end(),
                   expected_ambiguities.begin()));

  const parallel_phylogenetics::SequenceAlignment parsed_phylip =
      parallel_phylogenetics::ParsePhylip("3 3\nA ACG\nB_taxon A-N\nC TCG\n");
  Check(parsed_phylip.sites == 3);
  Check(parsed_phylip.records.size() == 3);
  Check(parsed_phylip.records[1].name == "B_taxon");
  Check(parsed_phylip.records[1].sequence == "A-N");

  const parallel_phylogenetics::SequenceAlignment interleaved_phylip =
      parallel_phylogenetics::ParsePhylip(
          "3 9\nA ACG\nB_taxon A-N\nC TCG\n\n"
          "  TTA\n  CCG\n  GGA\n\n  CCC\n  TTT\n  AAA\n");
  Check(interleaved_phylip.sites == 9);
  Check(interleaved_phylip.records[0].sequence == "ACGTTACCC");
  Check(interleaved_phylip.records[1].sequence == "A-NCCGTTT");
  Check(interleaved_phylip.records[2].sequence == "TCGGGAAAA");

  const parallel_phylogenetics::SequenceAlignment labelled_interleaved =
      parallel_phylogenetics::ParsePhylip("3 6\nA ACG\nB_taxon A-N\nC TCG\n\n"
                                          "A TTA\nB_taxon CCG\nC GGA\n");
  Check(labelled_interleaved.records[0].sequence == "ACGTTA");
  Check(labelled_interleaved.records[1].sequence == "A-NCCG");
  Check(labelled_interleaved.records[2].sequence == "TCGGGA");

  const parallel_phylogenetics::Phylogeny ambiguity_tree =
      parallel_phylogenetics::ParseNewick(
          "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
  const parallel_phylogenetics::SequenceAlignment ambiguity_alignment =
      parallel_phylogenetics::ParsePhylip(
          "4 12\nA ACRYSWKMBDHV\nB AAAAAAAAAAAA\n"
          "C TTTTTTTTTTTT\nD CCCCCCCCCCCC\n");
  const parallel_phylogenetics::EncodedAlignment ambiguity_encoded =
      parallel_phylogenetics::EncodeAlignment(ambiguity_tree,
                                              ambiguity_alignment);
  parallel_phylogenetics::SequentialWorkspace ambiguity_workspace;
  ambiguity_workspace.Reserve(ambiguity_tree.plan, ambiguity_encoded.sites);
  const std::span<const parallel_phylogenetics::Scalar> ambiguity_likelihoods =
      parallel_phylogenetics::LogLikelihoodsPrepared(
          {ambiguity_tree.plan, ambiguity_encoded.sites,
           ambiguity_tree.branch_lengths, ambiguity_encoded.observation_nodes,
           ambiguity_encoded.observations},
          ambiguity_workspace);
  // Independent per-site values from RAxML-NG under the same JC69 model.
  const std::array<parallel_phylogenetics::Scalar, 12>
      expected_ambiguity_likelihoods{
          -5.040094590240, -6.981834284265, -4.934072093490, -6.251378523747,
          -6.405406411110, -4.896548084956, -6.363574230408, -4.906040938581,
          -5.932518541162, -4.804068470622, -4.779412771342, -4.812719161660};
  Check(std::equal(ambiguity_likelihoods.begin(), ambiguity_likelihoods.end(),
                   expected_ambiguity_likelihoods.begin(),
                   [](double actual, double expected_value) {
                     return Near(actual, expected_value);
                   }));
}
