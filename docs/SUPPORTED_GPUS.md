# Supported GPUs

Nodexo currently accepts the GPU model/count configurations listed below.
Miner setup stops before registration when the detected GPU model or GPU count
is not covered by calibrated timing policy.

## Supported Today

| GPU model | VRAM | Accepted GPU count |
| --- | ---: | ---: |
| NVIDIA H100 80GB HBM3 | 80 GB | 1-2 |
| NVIDIA A100-SXM4-80GB | 80 GB | 1-4 |
| NVIDIA A100 80GB PCIe | 80 GB | 1 |
| NVIDIA RTX A6000 | 48 GB | 1 |
| NVIDIA GeForce RTX 4090 | 24 GB | 1 |

The live subnet config is authoritative. More GPU models and higher GPU-count
configurations will be added after reviewed calibration runs.
