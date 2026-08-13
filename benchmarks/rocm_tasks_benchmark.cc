#include "parallel_phylogenetics/rocm.h"

#define TASK_NAMESPACE parallel_phylogenetics::rocm
#define TASK_BACKEND_NAME "rocm"
#include "inference_tasks_benchmark.inc"
