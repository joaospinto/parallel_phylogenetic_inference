# Model support and package boundaries

## Where the richer phylogenetic model belongs

The three packages have deliberately different responsibilities:

1. `bidirectional_tree_rake_compress` (`btrc`) plans and replays topology
   operations. It never sees states, transition matrices, nucleotide models,
   or rate categories. HKY, GTR, and `+Gamma` require no scheduler change.
2. `parallel_tree_hmm` evaluates arbitrary finite-state node and edge factors.
   It already supports the four-state transition matrices produced by all of
   these nucleotide models and all forward and reverse inference tasks. It
   does not need biological model names.
3. `parallel_phylogenetic_inference` validates nucleotide parameters, builds
   stable finite-time transition matrices, creates discrete-Gamma or custom
   rate mixtures, maps IUPAC observations to categorical inputs, and combines
   category-conditional results. This is the only package changed for the new
   biological functionality.

For a rate mixture, the application currently retains one prepared generic
tree-HMM workspace per category. This preserves observation and factor
residency independently for every category and requires no category-specific
special case in the generic algebra. Likelihood and reverse outputs are exact:
the category is globally shared over the complete tree for a site, rather than
being selected independently on edges or nodes.

## What is now comparable with BEAGLE

Within four-state, stationary time-reversible nucleotide models, the library
now covers the common likelihood functionality represented by JC69, HKY, GTR,
and finite among-site rate mixtures such as discrete Gamma. It also exposes
node and edge marginals, joint MAP assignments, and exact posterior samples
through the same prepared accelerator interface—outputs beyond BEAGLE's usual
likelihood-oriented API.

This is not full feature parity with BEAGLE. Important BEAGLE capabilities not
implemented here include:

- arbitrary state counts and standard amino-acid or codon models;
- nonreversible models and general real/complex eigensystems;
- several simultaneous substitution models, partitions, and category schemes
  within one instance;
- an explicit zero-rate invariant-site category (`+I`);
- first and second transition-matrix derivatives, likelihood derivatives, and
  the broader gradient facilities introduced in BEAGLE 4.1;
- matrix convolution/epoch-model operations and its low-level user-managed
  buffer API;
- its mature collection of CPU SIMD, GPU, multi-device, and framework
  integrations.

Conversely, BEAGLE's public likelihood-oriented API does not provide this
project's topology-parallel contraction schedule and complete prepared APIs
for all ancestral node and edge marginals, joint MAP reconstruction, and exact
posterior sampling.
“Parity” is therefore meaningful only for the overlapping reversible
nucleotide likelihood calculation, not for the libraries as whole products.

## Why categories are not yet fused in `parallel_tree_hmm`

A fused helper API could represent `(site, rate category)` as one enlarged
batch while broadcasting a different edge-factor set per category. The
current generic batch deliberately broadcasts one edge-factor set, so such an
extension would change host layouts and every Metal, CUDA, and ROCm indexing
path. It would be an optimization, not a correctness requirement.

The present application-level implementation was chosen first because it is
small, auditable, preserves all three resident-input modes, and exercises the
same generic operations for every category. In a three-repeat diagnostic Apple
M4 FP32 run on a balanced 2,048-tip, 256-pattern GTR+4-Gamma problem, the
separate-category Metal implementation took 10.30 ms end to end versus 19.18
ms for the pinned ten-thread BEAGLE CPU adapter. BEAGLE recorded a 4.92 ms
pruning phase versus a 6.63 ms native device-event interval; those component
clocks are not directly matched, but they show that the complete-call advantage
does not establish a faster contraction kernel. BEAGLE's factor and input
paths made the complete call slower. A
single-rate cross-check took 4.80 ms for Metal and 5.97 ms for BEAGLE. These
diagnostics identify category fusion as a real kernel optimization opportunity
but do not justify immediately rewriting three GPU backends when the current
complete-call result is already competitive. A controlled CUDA/ROCm study can
decide whether launch/factor duplication dominates those platforms. The
public API does not preclude that later internal optimization.
