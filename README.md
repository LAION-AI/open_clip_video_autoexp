# LAION Big Video Dataset - ViCLIP experiments

[Project Page](https://projects.laion.ai/bvd/) · [Paper](https://arxiv.org/abs/2608.24845) · [Download](https://projects.laion.ai/bvd/download.html) · [GitHub](https://github.com/laion-ai/bvd)

## AutoExp config

For experiment management, we use autoexp <https://github.com/slampai/autoexperiment>.
We provide:

- the config file [config.yaml](config.yaml)
- train and eval template files [train.sbatch](train.sbatch) [eval.sbatch](eval.sbatch).

## Models

We provide our best ViCLIP L-14 model at <https://huggingface.co/laion/ViCLIP-L-14-BVD-V-50M-s50M-b32K-WiSE-FT>.

## Downstream tasks

The datasets used for evaluation (Kinetics-400, UCF-101, HMDB-51,  MSR-VTT, MSVD) are available at [https://huggingface.co/datasets/laion/video_benchmarks/tree/main](https://huggingface.co/datasets/laion/video_benchmarks/tree/main) in WebDataset form.

## Results

We provide full results on downstream tasks in [results.parquet](results.parquet) which can be displayed
using [summarize_results.py](summarize_results.py) helper script.

To show best results for each ViCLIP model size:

```
python summarize_results.py --model ViCLIP --keep_best
```

We also provide results with checkpoint merging:

```
pythonn summarize_wise_ft_results.py
```
