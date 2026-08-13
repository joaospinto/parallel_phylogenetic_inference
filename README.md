# Parallel Phylogenetic Inference

This repository implements phylogenetic likelihood and ancestral-state
inference on top of two independent reusable packages:

- [`parallel_tree_hmm`](https://github.com/joaospinto/parallel_tree_hmm)
  supplies prepared hidden Markov tree inference on CPU, Metal, CUDA, and
  ROCm;
- [`bidirectional_tree_rake_compress`](https://github.com/joaospinto/bidirectional_tree_rake_compress)
  supplies the topology planner and bidirectional rake–compress executor.

Phylogenetic models, alignments, and biological file formats live here. The
tree-HMM algebra and scheduler are not duplicated.

## Capabilities

The current implementation provides:

- JC69 likelihoods and log likelihoods on arbitrary rooted trees;
- node and edge posterior probabilities for individual sites;
- accelerator ancestral-state marginals, joint MAP assignments, and posterior
  samples for alignments;
- batched alignment conversion for the generic tree-HMM accelerator API;
- conventional scaled Felsenstein pruning as an independent CPU baseline;
- Newick and nucleotide FASTA input;
- preallocated numerical workspaces for repeated likelihood evaluations;
- phylogenetics-facing CUDA, ROCm, and Metal workspaces that select the
  appropriate tree-HMM representation and handle capacity-bounded site
  batches;
- native Metal, CUDA, and ROCm benchmark executables with numerical
  cross-checks.

The optional `beagle_benchmark` target compares the same JC69 likelihood with
the established BEAGLE library. BEAGLE is confined to benchmark code and is
not a dependency of the phylogenetic, tree-HMM, or rake--compress libraries.
The adapter requires the requested CPU or CUDA implementation and the same
FP32 or FP64 precision as this package; every result identifies the actual
BEAGLE resource and implementation selected at runtime.

Alignment conversion writes compact categorical observations directly into
caller-provided accelerator input storage. CUDA and ROCm use pinned host
buffers, while Metal uses shared buffers, so the same generic tree-HMM call
neither makes a second full-batch staging copy nor materializes dense node
factors on the host.

CPU, CUDA, and ROCm use the same compile-time `Scalar`: FP64 is the default,
and FP32 is a separate pure-precision build. Metal is FP32-only. Benchmark rows
report their precision and compare the CPU and accelerator implementations in
that same precision.

Applications do not need to prepare generic tree-HMM factors themselves. For
example, a prepared CUDA evaluation is:

```cpp
#include "parallel_phylogenetics/cuda.h"

parallel_phylogenetics::cuda::Workspace workspace;
workspace.Reserve(model, 4096);
std::span<const parallel_phylogenetics::Scalar> log_likelihoods =
    parallel_phylogenetics::cuda::LogLikelihoodsPrepared(model, workspace);
```

The ROCm interface differs only in its namespace. The corresponding Metal
interface also has no device argument. All three backends stage compact
categorical observations and call the same backend-neutral tree-HMM
contraction executor. Repeated prepared calls do not resize numerical
workspace storage or reconstruct the topology plan.

Prepared calls use the tree-HMM `CategoricalInputUpdate` lifecycle directly.
The default `kAll` stages observations and numerical factors. After one such
call, `kFactors` keeps the observations resident while updating the JC69 root,
emission, and edge factors; this is the relevant mode when tip data are fixed
but branch lengths or model parameters change. `kNone` reuses both observations
and factors for a genuinely fixed-model throughput measurement. `LastTimings()`
returns the summed backend transfer, kernel, download, and call times, the
application-level evaluation time, and the number of site batches.

Resident reuse applies to one exact site batch. If an alignment exceeds the
workspace capacity, call `SelectSites`, reserve and stage each selected batch
with `kAll`, and then measure that batch with `kFactors` or `kNone`. The
full-alignment convenience call deliberately rejects resident modes when it
would have to cycle different observations through one capacity-bounded
buffer; it never labels such transfers as resident execution.

Recovery uses operation-specific workspaces so likelihood-only evaluations do
not retain unnecessary reverse-pass data. For example, posterior marginals for
a bounded site batch are evaluated as follows:

```cpp
workspace.ReserveMarginals(model, 4096);
for (std::size_t first = 0; first < model.sites; first += 4096) {
  const auto batch = parallel_phylogenetics::SelectSites(
      model, first, std::min<std::size_t>(4096, model.sites - first));
  const parallel_phylogenetics::AlignmentPosteriorView posterior =
      parallel_phylogenetics::cuda::PosteriorMarginalsPrepared(batch,
                                                                workspace);
  // Consume this batch before reusing workspace.
}
```

`MaximumAPosterioriPrepared` has the same batching convention.
`PosteriorSamplePrepared` additionally accepts one caller-provided uniform
variate per site and node, making posterior draws reproducible without
embedding a random-number generator in the inference package.

## Build and test

Keep this repository beside `bidirectional_tree_rake_compress` and
`parallel_tree_hmm`. Bazel resolves both through local module overrides.

```sh
bazel test //... --config=fp64
```

On an Apple system with Metal:

```sh
bazel run //:metal_benchmark --config=fp32 -- \
  --topology balanced --leaves 4096 --sites 1024 --repeats 5
```

On a CUDA system, replace `sm_75` with the device's compute capability:

```sh
bazel run --config=fp64 --config=cuda --cuda_archs=sm_75 \
  //:cuda_benchmark -- \
  --topology balanced --leaves 4096 --sites 1024 --repeats 5
```

On Linux, one command validates the portable build policy. Bazel fetches
pinned CUDA 12.8.1 and ROCm 7.2.3 SDKs and compiles both backends; the driver
executes only the backend whose GPU is present. On macOS it builds and executes
Metal without fetching either Linux SDK.

```sh
scripts/accelerator_driver.sh
```

The defaults can be narrowed without editing the script, for example:

```sh
TREE_HMM_BUILD_BACKENDS=rocm TREE_HMM_RUN_BACKENDS=none \
  TREE_HMM_ROCM_ARCH=gfx942 scripts/accelerator_driver.sh
```

All three benchmark binaries accept empirical FASTA or relaxed sequential or
interleaved PHYLIP alignments:

```sh
bazel run //:metal_benchmark --config=fp32 -- \
  --newick family.nwk --fasta family.fasta --repeats 5
```

Long alignments can be evaluated in a capacity-bounded prepared workspace:

```sh
bazel run //:metal_benchmark --config=fp32 -- \
  --newick tree.nwk --phylip alignment.phy --site-batch 1024 --repeats 5
```

For a matched-precision CPU comparison with an installed BEAGLE library:

```sh
PRECISION=fp64 scripts/benchmark_beagle.sh \
  --newick family.nwk --fasta family.fasta \
  --beagle-resource cpu --beagle-threads 1 --repeats 5
```

On Linux, `scripts/install_beagle.sh` checksum-verifies and builds upstream
commit `d1e9c62f922cf544fda4555aedf113519367c07a`, identified as a BEAGLE
4.1.0 pre-release, for the CPU and, when requested, with its CUDA plugin. The
optional local dependency is discovered through `BEAGLE_PREFIX`; ordinary
Bazel targets remain buildable when BEAGLE is absent. Exact alternative source
commits can be selected with `BEAGLE_VERSION_LABEL`,
`BEAGLE_SOURCE_REVISION`, `BEAGLE_SOURCE_URL`, and
`BEAGLE_SOURCE_SHA256`; the installer records those values and its CMake flags
in `BEAGLE_BUILD_METADATA.txt`. A development branch is never silently
reported as a numbered release.

Warmup and workspace allocation are excluded. CPU and accelerator execution
order alternates between repetitions to reduce order bias. The standard Metal
sweep additionally performs five seconds of untimed interleaved conditioning
before each case to reduce thermal-state bias on passively cooled systems. The
reported total accelerator time includes conversion from the phylogenetic
model to generic tree-HMM factors, host/device transfer, kernel execution, and
result transfer.

The distribution study is a prespecified Cartesian product of taxa and unique
pattern counts over deterministic replicates of Yule, critical beta-splitting,
uniform/PDA, and caterpillar trees. Its synthetic patterns are distinct
performance inputs, not draws from JC69. Raw repeats and tree-shape, planning,
batch, and input-size metadata are retained in every CSV row:

```sh
PRECISION=fp32 scripts/benchmark_synthetic_distributions.sh cuda
PRECISION=fp32 TREE_HMM_BENCHMARK_MODE=factor-update \
  BEAGLE_THREADS=1 scripts/benchmark_synthetic_distributions.sh beagle-cpu
scripts/plot_synthetic_study.py native.log beagle.log \
  --native cuda --baseline beagle-cpu --precision FP32 \
  --benchmark-mode factor-update --run-identity same-machine-protocol \
  --output-directory figures
```

Native and BEAGLE rows distinguish `fixed-model` (all observations and numerical factors
already installed), `factor-update` (observations retained while JC69 factors
and transition matrices are refreshed), and `full-input-update` (tips and
factors refreshed). Pattern multiplicities are fixed compression metadata:
both timed methods produce per-unique-pattern log likelihoods, and the true
multiplicities are applied outside the timed region. The modes are never
combined into one speedup. Native resident timings stage each exact site chunk once with a
separately reported, untimed `full-input-update` call, then perform all timed
repetitions before reusing the workspace for another chunk; they do not imply
that a capacity-bounded full alignment is simultaneously device-resident.
BEAGLE receives unambiguous or missing tips in its compact-state form and uses
partial vectors only for genuinely ambiguous IUPAC tips. The exception is the
BEAGLE 4.1 pre-release CPU implementation during full-input or chunked runs:
those rows deliberately use reusable partial buffers because that version's
repeated compact-state setter allocates without reusing its prior buffer. Each
row records the compact and partial tip counts.
`cuda_tasks_benchmark`, `rocm_tasks_benchmark`, and
`metal_tasks_benchmark` separately time likelihoods, all node and edge
marginals, joint MAP assignments, and posterior samples through the public
prepared APIs.

## Public data

PANDIT 17.0 contains 7,738 protein-domain families with nucleotide alignments
and corresponding inferred phylogenies. To download the official archive,
verify its checksum, and extract DNA Newick/FASTA pairs:

```sh
scripts/fetch_pandit.sh
```

By default, columns marked unreliable by PANDIT's HMM mask are removed. The
extractor accepts family, leaf-count, and result-count filters; run
`scripts/extract_pandit.py --help` for details. Downloaded data are ignored by
Git and retain the terms of the [PANDIT data
license](https://www.ebi.ac.uk/goldman-srv/pandit/Pandit/data/COPYRIGHT).

`scripts/benchmark_pandit.sh` evaluates a deterministic subset selected from
the generated manifest. By default it includes every family with at least 100
tips; PANDIT 17.0 has 325 such families. It supports our Metal, CUDA, and ROCm
backends and BEAGLE's CPU or CUDA backend, and accepts environment variables
for the minimum tip count, repeat count, precision, and an optional smoke-test
limit.
`scripts/summarize_benchmarks.py` reads the CSV records embedded in the raw
logs, takes medians across repeated records, selects the fastest measured site
batch by a declared rule, and reports either the selected cases or paired
accelerator-versus-BEAGLE corpus distributions.  For example, the PANDIT
records embedded in a complete CUDA notebook report are summarized with

```sh
scripts/summarize_benchmarks.py report.txt \
  --dataset-prefix PF --precision FP32 \
  --benchmark-mode full-input-update --corpus cuda \
  --max-abs-error 0.1 --max-relative-error 0.001
```

Multiple input logs must carry the same notebook cache identity. For logs
captured outside that workflow, `--run-identity` is an explicit assertion that
the hardware and benchmark protocol match; data from different machines are
never pooled implicitly. Chunked resident-mode projections are excluded from
the publication summaries and plots.

The Fish Tree of Life supplies a much larger empirical nucleotide case with
11,638 taxa and 24,143 sites. `scripts/fetch_fish_tree.sh` downloads and
checksum-verifies its published RAxML tree and full alignment. The accelerator
notebook evaluates the complete alignment at several prepared site-batch
capacities; it does not truncate or replicate the data.

## Reproducible accelerator runs

`scripts/benchmark_metal.sh` runs the standard local Metal scaling sweep.
Optional environment flags add the synthetic distribution and reverse-task
studies, while `TREE_HMM_EMPIRICAL_MANIFEST` runs any prepared public corpus
through the same capacity-bounded manifest driver.
`notebooks/kaggle_accelerator_benchmark.ipynb` detects an NVIDIA or AMD GPU,
builds both Linux accelerator backends from checksum-pinned SDKs, executes the
backend matching the detected device, and records the host and device
configuration. CUDA execution is additionally checked with Compute Sanitizer.
Its default `curated` profile runs the complete large-data comparisons, a
25-family paired PANDIT cohort, and representative unchunked resident cases.
`TREE_HMM_BENCHMARK_PROFILE=complete` enables every precision, mode,
distribution replicate, task, and PANDIT family; individual section and mode
variables remain available as explicit overrides.
Before native execution, the launcher verifies that the host NVIDIA driver is
new enough for the pinned CUDA toolkit or that the pinned ROCm runtime can
enumerate the selected AMD device through the host kernel driver. The notebook
compares against the pinned BEAGLE 4.1.0 pre-release commit on the CPU and, on
NVIDIA, the CUDA device,
then evaluates the 325-family PANDIT subset and complete Fish Tree of Life
alignment in matched FP64 and FP32. For capacity-bounded alignment runs, each
method tries geometrically increasing site batches up to the complete
alignment. Device-allocation failures stop only the affected accelerator
sweep. CPU capacity probes run under a ceiling derived from the runtime memory
limit, so an oversized probe fails in its child process rather than invoking
the notebook-wide OOM killer. Unrelated failures remain fatal.

Until these repositories are public, create the notebook input bundle from
clean commits:

```sh
scripts/package_notebook_sources.sh
```

An interrupted run can be resumed without repeating completed configurations
by embedding its downloaded report in the next source bundle:

```sh
scripts/package_notebook_sources.sh \
  ~/worktrees/parallel_phylogenetics_corpora/parallel_tree_inference_sources.zip \
  previous_parallel_phylogenetics_cuda_report.txt
```

The notebook recognizes completed validation phases, individual benchmark
configurations, and PANDIT families from the earlier report and appends only
the missing results. Its standard run selects the `validation`, `synthetic`,
`fish`, and `pandit` sections. A notebook-side environment override can select
any subset without duplicating orchestration, for example
`TREE_HMM_BENCHMARK_SECTIONS="fish pandit"` together with
`TREE_HMM_PRECISIONS_OVERRIDE=FP32`. Existing `TREE_HMM_SKIP_*` controls,
repeat-count overrides, sanitizer controls, and memory-guard overrides remain
available.

For a prespecified broad empirical cohort,
`scripts/prepare_dryad_corpus.py` imports all 222 alignments in the pinned
archive's `empirical-long/dna_long_empirical` cohort from Dryad
10.5061/dryad.8gtht76zz. For each alignment it selects the maximum finite
log-likelihood row among `version == "standard"` in the corresponding
`pars_summary.parquet`, verifies exact tree/alignment taxon agreement, performs
exact duplicate-column compression, and records source and normalized hashes.
Records in the archive's separate `unsuccessful_MSAs` directory are not part of
this cohort. Selection is independent of benchmark timing. PyArrow is needed
only while creating this compact manifest; benchmark runtime remains
dependency-free.

The public TreeBASE Mirror used by the LvD study provides additional inferred
trees and their original alignments. The following workflow makes a blobless,
revision-pinned clone and prepares the first 400 eligible DNA pairs in a fixed
SHA-256 order. Only pairs with at least 100 exactly matching tree/alignment
taxa are eligible; every inspected exclusion and every source and normalized
hash is recorded.

```sh
scripts/clone_pinned_git_corpus.sh \
  https://github.com/angtft/TreeBASEMirror.git \
  c5bad4a1c3103244bc0d3a21db7a6b9329a9dc13 work/treebase-mirror
python3 scripts/prepare_treebase_mirror.py \
  work/treebase-mirror work/treebase-selected
```

RAxML Grove is an empirical *tree and fitted-metadata* database. Its authors
explicitly withhold the anonymized source alignments, so it cannot honestly be
treated as a corpus of empirical tree--alignment pairs. The corresponding
preparer instead selects 20 nucleotide trees by fixed hash rank in each of
five prespecified taxon-count bins and simulates 256 JC69 sites with recorded,
deterministic seeds. This provides realistic empirical topologies across a
wide size range while keeping the simulated observations clearly identified.

```sh
scripts/clone_pinned_git_corpus.sh \
  https://github.com/angtft/RAxMLGrove.git \
  b81faa13a93703fcdfbd6e6fa2ed1bb5b42b76f5 work/raxml-grove
python3 scripts/prepare_raxml_grove.py \
  work/raxml-grove work/raxml-grove-selected
```

No source or prepared corpus is distributed with this repository. Each
preparer writes `manifest.csv`, `excluded.csv`, and `corpus_metadata.txt`;
`scripts/benchmark_empirical_manifest.sh` consumes any such manifest and
copies its provenance into the benchmark log.

`scripts/prepare_ltplus.sh` checksum-verifies the official February 2026
LTPlus tree and alignment and invokes a two-pass streaming preparation. It
never materializes the 23 GB FASTA: the first pass records per-coordinate
coverage and observed bases, and the second emits positions with at least 50%
unambiguous coverage. Invariant columns are retained. To exercise the 261,845-
node topology without creating an 8+ GB derived alignment, the default output
is a deterministic, seed-recorded subset of at most 512 eligible coordinates;
pass `--maximum-coordinates 0` to retain all eligible coordinates. The exact
rule, selected-index hash, coverage distribution, input hashes, label
normalization, and one-to-one taxon check are recorded in the output manifest.
Exact duplicate patterns are compressed only after coordinate selection, with
their multiplicities retained in `pattern_weights.txt`.

No GitHub remote or push is required for that workflow.

## License

MIT
