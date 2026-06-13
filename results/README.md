# Arm64-cloud benchmark results

Produced by [`.github/workflows/arm-bench.yml`](../.github/workflows/arm-bench.yml) on a
free GitHub-hosted `ubuntu-24.04-arm` runner (4 vCPU aarch64, 15 GiB RAM, Azure westus2),
llama.cpp built with `-DGGML_CPU_KLEIDIAI=ON`, `gemma-4-12b-it-Q4_0` + `mtp-gemma-4-12b-it.gguf`.

- `arm64-ci-autotune.json` - draft-length sweep (best n-max=3, **2.02x**).
- `arm64-ci-bench-mtp.json` - MTP, n-max=3: vanilla 6.01 -> 12.05 tok/s (**2.00x**), 76% accept.
- `arm64-ci-bench-cascade.json` - MTP+n-gram cascade: 2.02x (a wash vs MTP-alone on Arm; 99.1% similarity).

Reproduce: Actions tab -> "arm64-cloud-benchmark" -> Run workflow (free, no credit card).
