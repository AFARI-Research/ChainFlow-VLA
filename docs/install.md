# Installation Guidance of ChainFlow-VLA

## 1. Create Conda Environment

```bash
cd /path/to/chainflow-vla
conda env create -f environment.yml
conda activate chainflow-vla
```

## 2. Prepare the NAVSIM Dataset

Please download and manage the NAVSIM dataset following the official NAVSIM installation guide:

[https://github.com/autonomousvision/navsim/blob/main/docs/install.md](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)

After the dataset is in place, point ChainFlow-VLA at it via the environment variables documented in `[docs/train_eval.md](train_eval.md)` — at minimum `OPENSCENE_DATA_ROOT` (the OpenScene/NAVSIM data root containing `navsim_logs/`, `sensor_blobs/`, and `meta_datas/`) and `NUPLAN_MAPS_ROOT` (the nuPlan map root).

## 3. Install the Repository

```bash
pip install -e ./nuplan-devkit --no-deps
pip install -e . --no-deps
```

Quick sanity check that the install succeeded:

```bash
python -c "import navsim, nuplan; print(navsim.__file__)"
```

## 4. Optional: Install FlashAttention for VLM Finetuning

FlashAttention is only required when reproducing the VLM finetuning stage adapted from RecogDrive. The detailed VLM finetuning pipeline follows RecogDrive, and users are encouraged to refer to the official RecogDrive repository for the complete finetuning instructions:

[https://github.com/xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive)

FlashAttention is used to reduce attention memory usage and improve training throughput for long visual/text contexts.

If you only use the released VLM features, train the action expert, or evaluate ChainFlow-VLA, this step can be skipped.

The tested environment uses:

- PyTorch: `2.5.1+cu124`
- CUDA: `12.4`
- Python: `3.9`
- Platform: Linux x86_64
- FlashAttention: `2.8.3`
- CXX11 ABI: `FALSE`

If `nvcc` is not available, install the CUDA 12.4 compiler/runtime inside the conda environment:

```bash
conda install -c nvidia cuda-nvcc=12.4 cuda-runtime=12.4 -y
```

Then set the CUDA paths:

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

which nvcc
nvcc --version
```

The corresponding wheel is:

```text
flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

You can download and install it with:

```bash
wget -O flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

pip install ./flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

If the wheel is already available locally, install it directly:

```bash
pip install /path/to/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

Please make sure that the FlashAttention wheel matches your Python version, PyTorch version, CUDA version, platform, and CXX11 ABI. Otherwise, installation may fail or the package may not be importable.