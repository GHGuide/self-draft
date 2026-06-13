# Arm64-cloud benchmark results

Produced by [`.github/workflows/arm-bench.yml`](../.github/workflows/arm-bench.yml) on a
free GitHub-hosted `ubuntu-24.04-arm` runner (4 vCPU aarch64, 15 GiB RAM, Azure westus2),
llama.cpp built with `-DGGML_CPU_KLEIDIAI=ON`, `gemma-4-12b-it-Q4_0` + `mtp-gemma-4-12b-it.gguf`.

- `arm64-ci-autotune.json` - draft-length sweep (best n-max=3, **2.02x**).
- `arm64-ci-bench.json` - n-max=3: vanilla 5.99 -> 12.01 tok/s (**2.0x**), latency
  36.2s -> 19.3s (1.87x lower), draft acceptance 76%, output similarity 99.1%.

Reproduce: Actions tab -> "arm64-cloud-benchmark" -> Run workflow (free, no credit card).
