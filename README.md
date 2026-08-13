# Parallel Phylogenetic Inference

This repository implements phylogenetic likelihood and ancestral-state
inference on top of two independent reusable packages:

- [`parallel_tree_hmm`](https://github.com/joaospinto/parallel_tree_hmm)
  supplies prepared hidden Markov tree inference on CPU, Metal, and CUDA;
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
- phylogenetics-facing CUDA and Metal workspaces that select the appropriate
  tree-HMM representation and handle capacity-bounded site batches;
- native Metal and CUDA benchmark executables with numerical cross-checks.

The optional `beagle_benchmark` target compares the same JC69 likelihood with
the established BEAGLE library. BEAGLE is confined to benchmark code and is
not a dependency of the phylogenetic, tree-HMM, or rake--compress libraries.
The adapter requires the requested CPU or CUDA implementation and the same
FP32 or FP64 precision as this package; every result identifies the actual
BEAGLE resource and implementation selected at runtime.

Alignment conversion writes compact categorical observations directly into
caller-provided accelerator input storage. CUDA uses pinned host buffers and
Metal uses shared buffers, so the same generic tree-HMM call neither makes a
second full-batch staging copy nor materializes dense node factors on the host.

CPU and CUDA use the same compile-time `Scalar`: FP64 is the default, and FP32
is a separate pure-precision build. Metal is FP32-only. Benchmark rows report
their precision and compare the CPU and accelerator implementations in that
same precision.

Applications do not need to prepare generic tree-HMM factors themselves. For
example, a prepared CUDA evaluation is:

```cpp
#include "parallel_phylogenetics/cuda.h"

parallel_phylogenetics::cuda::Workspace workspace;
workspace.Reserve(model, 4096);
std::span<const parallel_phylogenetics::Scalar> log_likelihoods =
    parallel_phylogenetics::cuda::LogLikelihoodsPrepared(model, workspace);
```

The corresponding Metal interface differs only in the namespace and has no
device argument. Both backends stage compact categorical observations and call
the same backend-neutral tree-HMM contraction executor. Repeated prepared
calls do not resize numerical workspace storage or reconstruct the topology
plan.

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

Both benchmark binaries accept empirical FASTA or relaxed sequential PHYLIP
alignments:

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

On Linux, `scripts/install_beagle_cuda.sh` checksum-verifies and builds the
pinned BEAGLE 4.0.1 release with its CUDA plugin. The optional local dependency
is discovered through `BEAGLE_PREFIX`; ordinary Bazel targets remain buildable
when BEAGLE is absent.

Warmup and workspace allocation are excluded. CPU and accelerator execution
order alternates between repetitions to reduce order bias. The standard Metal
sweep additionally performs five seconds of untimed interleaved conditioning
before each case to reduce thermal-state bias on passively cooled systems. The
reported total accelerator time includes conversion from the phylogenetic
model to generic tree-HMM factors, host/device transfer, kernel execution, and
result transfer.

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
tips; PANDIT 17.0 has 325 such families. It supports our Metal or CUDA backend
and BEAGLE's CPU or CUDA backend, and accepts environment variables for the
minimum tip count, repeat count, precision, and an optional smoke-test limit.
`scripts/summarize_benchmarks.py` reads the CSV records embedded in the raw
logs, takes medians across repeated records, selects the fastest measured site
batch by a declared rule, and reports either the selected cases or paired
accelerator-versus-BEAGLE corpus distributions.  For example, the PANDIT
records embedded in a complete CUDA notebook report are summarized with

```sh
scripts/summarize_benchmarks.py report.txt \
  --dataset-prefix PF --precision FP32 --corpus cuda
```

The Fish Tree of Life supplies a much larger empirical nucleotide case with
11,638 taxa and 24,143 sites. `scripts/fetch_fish_tree.sh` downloads and
checksum-verifies its published RAxML tree and full alignment. The CUDA
notebook evaluates the complete alignment at several prepared site-batch
capacities; it does not truncate or replicate the data.

## Reproducible accelerator runs

`scripts/benchmark_metal.sh` runs the standard local scaling sweep. For CUDA,
`notebooks/kaggle_cuda_benchmark.ipynb` records machine information, executes
the host and emulated-kernel tests, validates the native CUDA implementation
with Compute Sanitizer, and then runs the same balanced and caterpillar sweep.
It additionally compares against pinned BEAGLE 4.0.1 on the CPU and CUDA
device and evaluates the 325-family PANDIT subset and complete Fish Tree of
Life alignment in matched FP64 and FP32. For capacity-bounded alignment runs,
each method tries geometrically increasing site batches up to the complete
alignment. Device-allocation failures stop only the affected CUDA sweep. CPU
capacity probes run under a ceiling derived from the runtime memory limit, so
an oversized probe fails in its child process rather than invoking the
notebook-wide OOM killer. Unrelated failures remain fatal.

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

No GitHub remote or push is required for that workflow.

## License

MIT
