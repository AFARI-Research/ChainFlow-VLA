<div align="center">
<img src="assets/chainflow_animated.gif" width="600">

<h1>  ChainFlow-VLA: Causal Flow Planning with Vision-Language Models </h1>

<strong>Xiyang Wang<sup>1</sup></strong><sup>\*</sup>, <strong>Xinlin Wang<sup>1</sup></strong><sup>\*</sup>, <strong>Tingguang Zhou<sup>1</sup></strong><sup>\*</sup>, <strong>Gong Chen<sup>1,2</sup></strong><sup>\*</sup>, <strong>Xingtai Gui<sup>1,3</sup></strong>, <strong>Zhi Xu<sup>1</sup></strong>, <strong>Xiaolei Wu<sup>1</sup></strong>, <strong>Feiyang Tan<sup>1</sup></strong>, <strong>Hangning Zhou<sup>1,†,✉</sup></strong>, <strong>Mu Yang<sup>1</sup></strong>

<sup>1</sup>Afari Intelligent Drive &nbsp;&nbsp; <sup>2</sup>Tianjin University &nbsp;&nbsp; <sup>3</sup>University of Macau

*Equal contribution. Listed in no particular order.    †Project Leader.  
<sup>✉</sup> Corresponding author (zhouhangning@qianli-drive.com).

[![NAVSIM v1 Leaderboard #1](https://img.shields.io/badge/🏆%20NAVSIM%20v1-TOP%201-C9A227?style=flat)](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)
[![Paper PDF](https://img.shields.io/badge/arXiv-ChainFlowVLA-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2605.23270)
[![Huggingface Weights](https://img.shields.io/badge/Weights-ChainFlowVLA-2C5282?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/AFARI-Research/ChainFlow-VLA/tree/main/weights)
[![Huggingface Datasets](https://img.shields.io/badge/Datasets-ChainFlowVLA-319795?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/AFARI-Research/ChainFlow-VLA-VLM-Feature-Cache/tree/main)
</div>

## News
- `[2026/07/10]` The repository is now open-sourced and publicly available on GitHub.
- `[2026/06/03]` Cached VLM features were released at [Huggingface](https://huggingface.co/datasets/AFARI-Research/ChainFlow-VLA-VLM-Feature-Cache/tree/main) and [ModelScope](https://modelscope.cn/datasets/AFARI/ChainFlow-VLA-VLM-Feature-Cache).
- `[2026/06/02]` Model weights were released at [Huggingface](https://huggingface.co/AFARI-Research/ChainFlow-VLA/tree/main/weights) and [Modelscope](https://modelscope.cn/models/AFARI/ChainFlow-VLA).
- `[2026/05/22]` We released our paper on [arXiv](https://arxiv.org/pdf/2605.23270).
- `[2026/05/04]` ChainFlow-VLA ranked **TOP 1** on the [NAVSIM v1 leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest), reaching human-level performance and surpassing the human reference score.

## Overview
ChainFlow-VLA is a unified vision-language-action framework for end-to-end autonomous driving. Instead of treating causal reasoning and global optimization as separate planning paradigms, it formulates trajectory prediction as a single Chain-to-Flow process:

- `Chain`: an autoregressive generator first proposes a small set of causally consistent trajectory modes.
- `Flow`: a diffusion refiner then performs residual correction in trajectory space for better global structure and robustness.
- `VLA`: hidden states from a vision-language model are injected as semantic flow conditioning, enabling scene-aware refinement in ambiguous and long-tail scenarios.

On the NAVSIM v1 benchmark, ChainFlow-VLA reaches **94.85 PDMS**, achieving state-of-the-art performance at human-level quality.

## Visualization

The BEV rollouts below compare NAVSIM human expert rollouts on the left with ChainFlow-VLA rollouts on the right for the same representative scenes.


| Human Expert | ChainFlow-VLA |
| :----------: | :-----------: |
| <img src="assets/GT_020dee65dab453bb_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_020dee65dab453bb_BEV_future_agents.gif" width="380"/> |
| <img src="assets/GT_0eb7dda83bbe5fb2_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_0eb7dda83bbe5fb2_BEV_future_agents.gif" width="380"/> |
| <img src="assets/GT_277f191c94b952f3_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_277f191c94b952f3_BEV_future_agents.gif" width="380"/> |
| <img src="assets/GT_4376d00ed2245c21_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_4376d00ed2245c21_BEV_future_agents.gif" width="380"/> |
| <img src="assets/GT_923e4fcf3daa57f8_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_923e4fcf3daa57f8_BEV_future_agents.gif" width="380"/> |
| <img src="assets/GT_a2d180a344d15054_BEV_gt.gif" width="380"/> | <img src="assets/ChainFlow_a2d180a344d15054_BEV_future_agents.gif" width="380"/> |


Across these cases, ChainFlow-VLA shows strong lane-level alignment and coherent interactions with nearby agents.

## Getting Started

- [Preparation of ChainFlow-VLA environment](docs/install.md)
- [Training and evaluation](docs/train_eval.md)

## Checkpoint

> Results on NAVSIM


| Method        | Dataset Split | PDMS | Weight Download                                                                          |
| ------------- | ------------- | ---- | ---------------------------------------------------------------------------------------- |
| ChainFlow-VLA | Train 85k     | 93.8 | [Hugging Face](https://huggingface.co/AFARI-Research/ChainFlow-VLA/tree/main/weights/stage_2) / [Model Scope](https://modelscope.cn/models/AFARI/ChainFlow-VLA/tree/master/weights/stage_2) |
| ChainFlow-VLA | Trainval 103k | 94.8 | [Hugging Face](https://huggingface.co/AFARI-Research/ChainFlow-VLA/tree/main/weights/stage_2) / [Model Scope](https://modelscope.cn/models/AFARI/ChainFlow-VLA/tree/master/weights/stage_2) |


## BibTeX

```bibtex
@misc{wang2026chainflowvlacausalflowplanning,
      title={ChainFlow-VLA: Causal Flow Planning with Vision-Language Models}, 
      author={Xiyang Wang and Xinlin Wang and Tingguang Zhou and Gong Chen and Xingtai Gui and Zhi Xu and Xiaolei Wu and Feiyang Tan and Hangning Zhou and Mu Yang},
      year={2026},
      eprint={2605.23270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.23270}, 
}
```

## Acknowledgement

We thank the NAVSIM benchmark team and the broader autonomous driving research community for releasing datasets, evaluation tools, and strong open baselines that make this line of research possible.

We also acknowledge inspiring prior projects including [ReCogDrive](https://github.com/xiaomi-research/recogdrive) and [DrivoR](https://github.com/valeoai/DrivoR/).