# LAION Big Video Dataset - ViCLIP experiments

[Project Page](https://projects.laion.ai/bvd/) · [Paper](https://arxiv.org/abs/2608.24845) · [Download](https://projects.laion.ai/bvd/download.html) · [GitHub](https://github.com/laion-ai/bvd)

## AutoExp config

For experiment management, we use autoexp <https://github.com/slampai/autoexperiment>.
We provide the config [config.yaml](config.yaml), and template files [train.sbatch](train.sbatch),
[eval.sbatch](eval.sbatch).


## Results

We provide full results on downstream tasks in [results.parquet](results.parquet) which can be displayed
using [summarize_results.py](summarize_results.py) helper script.
The datasets used for evaluation (Kinetics-400, UCF-101, HMDB-51,  MSR-VTT, MSVD) are available at <https://huggingface.co/laion/video_benchmarks>
in WebDataset form.


To show best results for each ViCLIP model size:

```
python summarize_results.py --model ViCLIP --keep_best
```

We also provide results with checkpoint merging:

```
pythonn summarize_wise_ft_results.py
```
