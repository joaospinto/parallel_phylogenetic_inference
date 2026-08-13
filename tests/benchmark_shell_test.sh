#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
else
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
report="${root}/tests/benchmark_report_fixture.txt"
# shellcheck source=scripts/benchmark_resume.sh
source "${root}/scripts/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${root}/scripts/capacity_bounded.sh"

benchmark_resume_case_completed "${report}" cuda FP64 synthetic balanced \
  256 1024 1024
! benchmark_resume_case_completed "${report}" cuda FP32 synthetic balanced \
  256 1024 1024
benchmark_resume_case_completed "${report}" beagle-cpu FP64 synthetic \
  caterpillar 64 256 256
benchmark_resume_dataset_batch_completed "${report}" cuda FP64 \
  actinopt_12k_raxml 2048
! benchmark_resume_dataset_batch_completed "${report}" cuda FP64 \
  actinopt_12k_raxml 4096
benchmark_resume_dataset_completed "${report}" metal FP32 PF00004
benchmark_resume_capacity_reached "${report}" cuda FP64 4096
benchmark_resume_capacity_reached "${report}" cuda FP64 8192
! benchmark_resume_capacity_reached "${report}" cuda FP64 2048
benchmark_resume_validation_completed "${report}" FP32
benchmark_resume_validation_completed "${report}" FP64
! benchmark_resume_validation_completed "${report}" FP16

work_directory="$(mktemp -d "${TMPDIR:-/tmp}/benchmark-shell-test.XXXXXX")"
trap 'rm -rf "${work_directory}"' EXIT
benchmark_capacity_exhausted=0
success_output="${work_directory}/success.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 1 \
  bash -c 'printf success' > "${success_output}"
[[ "$(<"${success_output}")" == success ]]
[[ "${benchmark_capacity_exhausted}" == 0 ]]

benchmark_capacity_exhausted=0
allocation_output="${work_directory}/allocation.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 2 \
  bash -c 'echo std::bad_alloc >&2; exit 1' > "${allocation_output}" 2>&1
grep -Fq "first_infeasible_site_batch=2" "${allocation_output}"
[[ "${benchmark_capacity_exhausted}" == 1 ]]

benchmark_capacity_exhausted=0
killed_output="${work_directory}/killed.txt"
benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 4 \
  bash -c 'kill -9 $$' > "${killed_output}" 2>&1
grep -Fq "first_infeasible_site_batch=4" "${killed_output}"
[[ "${benchmark_capacity_exhausted}" == 1 ]]

guard_output="$(benchmark_run_capacity_bounded "${work_directory}" 262144 \
  beagle-cpu FP32 8 bash -c 'ulimit -v')"
[[ "${guard_output}" == 262144 ]]

if benchmark_run_capacity_bounded "${work_directory}" 262144 cuda FP32 16 \
  bash -c 'exit 42'; then
  echo "an unrelated failure was misclassified as a capacity limit" >&2
  exit 1
else
  status=$?
  [[ "${status}" == 42 ]]
fi

launcher_root="${work_directory}/launcher"
source_root="${launcher_root}/input/sources"
working_root="${launcher_root}/working"
mkdir -p "${source_root}/parallel_phylogenetic_inference/scripts" \
  "${source_root}/parallel_tree_hmm" \
  "${source_root}/bidirectional_tree_rake_compress" "${working_root}"
cp "${root}/scripts/kaggle_cuda_notebook.sh" \
  "${source_root}/parallel_phylogenetic_inference/scripts/"
cat > "${source_root}/parallel_phylogenetic_inference/scripts/notebook_cuda.sh" <<'EOF'
#!/usr/bin/env bash
printf 'stub precision=%s skip_fish=%s\n' \
  "${TREE_HMM_PRECISIONS}" "${TREE_HMM_SKIP_FISH_TREE:-0}"
printf 'stub sections=%s\n' "${TREE_HMM_BENCHMARK_SECTIONS:-}"
EOF
printf 'previous-row\n' > "${source_root}/PREVIOUS_BENCHMARK_REPORT.txt"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP32 TREE_HMM_SKIP_FISH_TREE=1 \
TREE_HMM_BENCHMARK_SECTIONS="fish pandit" \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_root}" \
  > "${launcher_root}/unpacked.log"
grep -Fq 'using unpacked source input' "${launcher_root}/unpacked.log"
grep -Fq 'previous-row' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub precision=FP32 skip_fish=1' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
grep -Fq 'stub sections=fish pandit' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"

source_zip="${launcher_root}/parallel_tree_inference_sources.zip"
(
  cd "${source_root}"
  zip -q -r "${source_zip}" .
)
rm -rf "${working_root}/parallel-tree-inference"
TREE_HMM_NOTEBOOK_INPUT_DIR="${launcher_root}/input" \
TREE_HMM_NOTEBOOK_WORKING_DIR="${working_root}" \
TREE_HMM_PRECISIONS_OVERRIDE=FP64 TREE_HMM_SKIP_FISH_TREE=0 \
  bash "${root}/scripts/kaggle_cuda_notebook.sh" "${source_zip}" \
  > "${launcher_root}/zip.log"
grep -Fq 'source bundle SHA-256:' "${launcher_root}/zip.log"
grep -Fq 'stub precision=FP64 skip_fish=0' \
  "${working_root}/parallel_phylogenetics_cuda_report.txt"
