#include "parallel_phylogenetics/cuda.h"

#define PARALLEL_PHYLOGENETICS_ACCELERATOR_NAMESPACE                         \
  parallel_phylogenetics::cuda
#define PARALLEL_PHYLOGENETICS_GPU_NAME "CUDA"
#define PARALLEL_PHYLOGENETICS_BACKEND_NAME "cuda"
#include "benchmarks/gpu_backend_benchmark.inc"
#undef PARALLEL_PHYLOGENETICS_BACKEND_NAME
#undef PARALLEL_PHYLOGENETICS_GPU_NAME
#undef PARALLEL_PHYLOGENETICS_ACCELERATOR_NAMESPACE
