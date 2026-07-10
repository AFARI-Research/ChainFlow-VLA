# Training, Evaluation, and Submission

This document describes the training, evaluation, checkpoint-watching, and submission scripts for ChainFlow-VLA. The runnable entry points live under `scripts/{training,evaluation,submission}/`.

The shell variables exposed by these scripts are limited to runtime essentials: data paths, output paths, checkpoint paths, split names, GPU/data-loader sizes, and optional Hydra overrides. Model behavior such as diffusion usage, scorer training, VLM cache format/name, hidden-state key, diffusion inference steps, and eval noise seed is controlled by the selected YAML config under `navsim/planning/script/config/`.

## Common Paths

Set the common paths before running any of the scripts below:

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/ChainFlow-VLA
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NUPLAN_MAPS_ROOT=/path/to/maps
export NAVSIM_EXP_ROOT=/path/to/experiments
export NAVSIM_METRIC_CACHE_ROOT=/path/to/metric_cache_root
export PYTHON_BIN=/path/to/miniconda3/envs/chainflow-vla/bin/python
```

## Training

Cache train metrics for pdm score calculation:

```
python navsim/planning/script/run_train_metric_caching.py
```

Train Stage 1 (`navtrainval`)

```bash
AGENT=chainflow_vla_stage1 \
EXPERIMENT_NAME=stage1_trainval \
TRAIN_TEST_SPLIT=navtrainval \
GPUS_PER_NODE=8 \
bash scripts/training/train_chainflow_vla.sh
```

Train Stage 2 from a released Stage 1 checkpoint (`navtrainval`):

```bash
AGENT=chainflow_vla_stage2 \
EXPERIMENT_NAME=stage2_trainval \
TRAIN_TEST_SPLIT=navtrainval \
TRAIN_PRETRAIN_CKPT_PATH=/path/to/stage1_trainval.ckpt \
VLM_FEATURE_CACHE_PATH=/path/to/navtrain_vlm_feature_cache \
GPUS_PER_NODE=8 \
bash scripts/training/train_chainflow_vla.sh
```

To train on `navtrain` only, change `TRAIN_TEST_SPLIT=navtrain` and pick a matching `EXPERIMENT_NAME`. For Stage 2, also point `TRAIN_PRETRAIN_CKPT_PATH` at a Stage 1 checkpoint trained on the same split.

## VLM Finetuning and Feature Cache

VLM finetuning follows the RecogDrive-style InternVL finetuning pipeline. As described in [install.md](install.md), FlashAttention is only required for reproducing this VLM finetuning stage. If you only use released VLM features, train the ChainFlow-VLA action expert, or evaluate ChainFlow-VLA, the FlashAttention installation step can be skipped.

Most users do **not** need to finetune the VLM themselves. We recommend, in order of preference:

1. **Use the released VLM feature cache** (recommended). Download from [Huggingface](https://huggingface.co/datasets/AFARI-Research/ChainFlow-VLA-VLM-Feature-Cache/tree/main) or [ModelScope](https://modelscope.cn/datasets/AFARI/ChainFlow-VLA-VLM-Feature-Cache) (for users from China) and point Stage 2 training / evaluation at it via `VLM_FEATURE_CACHE_PATH`.
2. **Build your own cache from RecogDrive's released weights**. Run `scripts/cache_dataset/cache_internvl_single_node.sh` with `MODEL_PATH` set to the official RecogDrive VLM [checkpoint](https://huggingface.co/owl10/ReCogDrive-VLM-2B).
3. **Finetune the VLM from scratch** (only if you want to deviate from the released recipe).

VLM finetuning typically requires multi-node or multi-GPU setups. We ran our experiments using 8x A800 80GB GPUs on a single node. To start the single-node training, run:

```bash
bash scripts/training/finetune_internvl_2B_single_node.sh
```

To test direct trajectory prediction from the VLM agent:

```bash
bash scripts/evaluation/run_internvl_agent_pdm_score_evaluation.sh
```

To cache VLM `last_hidden_state` features for ChainFlow-VLA training:

```bash
bash scripts/cache_dataset/cache_internvl_single_node.sh
```

## Evaluation

Cache the navtest PDM metrics before the first evaluation (one-time cost; the cache is reused by every subsequent eval run):

```
bash scripts/evaluation/run_metric_caching.sh
```

Run PDM evaluation (`navtest`):

```bash
CHECKPOINT_PATH=/path/to/stage2.ckpt \
AGENT=chainflow_vla_stage2 \
EXPERIMENT_NAME=stage2_navtest \
TRAIN_TEST_SPLIT=navtest \
METRIC_CACHE_PATH=$NAVSIM_METRIC_CACHE_ROOT/metric_cache \
VLM_FEATURE_CACHE_PATH=/path/to/navtest_vlm_feature_cache \
GPUS_PER_NODE=8 \
bash scripts/evaluation/run_chainflow_vla_pdm_score_evaluation.sh
```

## Watch Checkpoints

Watch a training directory and evaluate each new checkpoint:

```bash
CHECKPOINT_DIR=$NAVSIM_EXP_ROOT/chainflow_vla/stage2_trainval \
AGENT=chainflow_vla_stage2 \
TRAIN_TEST_SPLIT=navtest \
METRIC_CACHE_PATH=$NAVSIM_METRIC_CACHE_ROOT/metric_cache \
VLM_FEATURE_CACHE_PATH=/path/to/navtest_vlm_feature_cache \
EVAL_SAVE_NAME=eval_12step \
GPUS_PER_NODE=8 \
bash scripts/evaluation/watch_chainflow_vla_ckpt_and_eval.sh
```

## Visualization

ChainFlow-VLA ships two complementary visualization paths under `navsim/evaluate/`:

- **Model-trajectory rendering** — `navsim/evaluate/render_top1_from_detail.py` takes a `proposal_top1/detail.json` written by the eval pipeline and produces BEV + front-camera figures (and, optionally, a BEV GIF with future GT agents) for a single token. For batch rendering of the ChainFlow-VLA qualitative token set, use the wrapper `navsim/evaluate/chainflow_vis.sh` (set `BASE`, `OPENSCENE_DATA_ROOT`, and `NUPLAN_MAPS_ROOT`; see the header of the script for all knobs).
- **GT-only BEV GIFs** — `navsim/evaluate/render_gt_gif_from_tokens.py` is a CLI/API renderer that writes a green-trajectory GT GIF for any `(token, log_name)` pair. The companion wrapper `scripts/viz/render_gt_gif_pairs.py` renders the fixed qualitative pair list in one call.

A walkthrough of the lower-level plotting primitives (`navsim/visualization/*.py`) and how to stitch them into custom figures or GIFs lives in [`tutorial/tutorial_visualization.ipynb`](../tutorial/tutorial_visualization.ipynb).

## Submission

Create a submission pickle. The `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` fields are embedded verbatim into the pickle and shown on the leaderboard — replace the placeholders with real values before submitting:

```bash
TEAM_NAME=your_team \
AUTHORS=author_1,author_2 \
EMAIL=contact@example.com \
INSTITUTION=your_institution \
COUNTRY=your_country \
CHECKPOINT_PATH=/path/to/stage2.ckpt \
AGENT=chainflow_vla_stage2 \
TRAIN_TEST_SPLIT=private_test_e2e \
VLM_FEATURE_CACHE_PATH=/path/to/private_test_vlm_feature_cache \
SUBMISSION_OUTPUT_DIR=$NAVSIM_EXP_ROOT/submission/stage2_private_test \
GPUS_PER_NODE=8 \
bash scripts/submission/run_chainflow_vla_create_submission_pickle.sh
```

The submission script writes `submission.pkl` under `SUBMISSION_OUTPUT_DIR`.

## Extra Overrides

For uncommon changes, pass Hydra overrides through one of the extra override variables instead of adding new shell variables:

```bash
TRAIN_EXTRA_OVERRIDES='agent.config.some_training_option=value'
EVAL_EXTRA_OVERRIDES='agent.config.some_eval_option=value'
SUBMISSION_EXTRA_OVERRIDES='agent.config.some_submission_option=value'
```

## Runtime Variables

- `NAVSIM_DEVKIT_ROOT`: repository root.
- `OPENSCENE_DATA_ROOT`: OpenScene/NAVSIM data root.
- `NUPLAN_MAPS_ROOT`: nuPlan map root.
- `NAVSIM_EXP_ROOT`: experiment output root.
- `NAVSIM_METRIC_CACHE_ROOT`: metric cache root.
- `PYTHON_BIN`: Python executable used by the scripts.
- `AGENT`: selected Hydra agent config, for example `chainflow_vla_stage2`.
- `EXPERIMENT_NAME`: experiment or evaluation output name.
- `TRAIN_TEST_SPLIT`: dataset split, for example `navtrainval`, `navtrain` or `navtest`.
- `TRAIN_PRETRAIN_CKPT_PATH`: Stage 1 checkpoint used to initialize Stage 2 training.
- `CHECKPOINT_PATH`: checkpoint used for evaluation or submission.
- `CHECKPOINT_DIR`: directory watched by `watch_chainflow_vla_ckpt_and_eval.sh`.
- `METRIC_CACHE_PATH`: metric cache used for PDM evaluation.
- `VLM_FEATURE_CACHE_PATH`: VLM feature cache directory.
- `SUBMISSION_OUTPUT_DIR`: output directory for `submission.pkl`.
- `GPUS_PER_NODE`: number of local GPUs used by the script.
- `TRAIN_EXTRA_OVERRIDES`: extra Hydra overrides for training.
- `EVAL_EXTRA_OVERRIDES`: extra Hydra overrides for evaluation and checkpoint watching.
- `SUBMISSION_EXTRA_OVERRIDES`: extra Hydra overrides for submission generation.

