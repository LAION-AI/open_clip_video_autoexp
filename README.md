# LAION Big Video Dataset - ViCLIP experiments

[Project Page](https://projects.laion.ai/bvd/) · [Paper](https://arxiv.org/abs/2608.24845) · [Download](https://projects.laion.ai/bvd/download.html) · [GitHub](https://github.com/laion-ai/bvd)

## AutoExp config

For experiment management, we use autoexp <https://github.com/slampai/autoexperiment>.
We provide the config [config.yaml](config.yaml), and template files [train.sbatch](train.sbatch),
[eval.sbatch](eval.sbatch).

## Results

We provide full results on downstream tasks in [results.parquet](results.parquet) which can be displayed
using [summarize_results.py](summarize_results.py) helper script.

Best results for each ViCLIP model size can be displayed using:

```
python summarize_results.py --model ViCLIP --keep_best
```

MSR-VTT_image_retrieval_recall@1_test MSR-VTT_text_retrieval_recall@1_test MSVD_image_retrieval_recall@1_test MSVD_text_retrieval_recall@1_test avg_overall_test
| model | global_batch_size | pretrain_dataset | samples_seen | lr | flops | kinetics400_acc1_test | UCF-101-with-splits_acc1_test | hmdb51-with-splits_acc1_test | MSR-VTT_image_retrieval_recall@1_test | MSR-VTT_text_retrieval_recall@1_test | MSVD_image_retrieval_recall@1_test | MSVD_text_retrieval_recall@1_test | avg_overall_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViCLIP_ViT-B-16 | 32000 | bvd_10m | 50M | 0.0004 | 1.4e19 | 1.8 | 5.2 | 4.5 | 0.1 | 0.6 | 0.2 | 0.5 | 2.1 |
| ViCLIP_ViT-B-16 | 32000 | bvd_50m | 50M | 4e-05 | 1.4e19 | 1.8 | 5.2 | 4.5 | 0.1 | 0.6 | 0.2 | 0.5 | 2.1 |
| ViCLIP_ViT-B-32 | 32000 | bvd_55m | 10M | 0.0001 | 7.4e17 | 49.9 | 70.2 | 44.6 | 34.2 | 38.1 | 44.2 | 74.8 | 51.4 |
| ViCLIP_ViT-B-32 | 32000 | bvd_55m | 50M | 0.0001 | 3.7e18 | 51.6 | 70.4 | 45.4 | 35.6 | 39.0 | 46.3 | 77.7 | 52.7 |
| ViCLIP_ViT-B-16 | 32000 | bvd_55m | 10M | 0.0001 | 2.7e18 | 54.6 | 71.8 | 51.7 | 36.0 | 40.0 | 48.6 | 79.0 | 55.1 |
| ViCLIP_ViT-B-16 | 32000 | bvd_10m | 10M | 0.0001 | 2.7e18 | 54.4 | 74.4 | 49.9 | 37.2 | 39.6 | 48.9 | 77.2 | 55.2 |
| ViCLIP_ViT-B-16 | 32000 | bvd_50m | 10M | 0.0001 | 2.7e18 | 54.3 | 73.2 | 50.9 | 37.4 | 40.5 | 49.3 | 78.6 | 55.5 |
| ViCLIP_ViT-B-16 | 32000 | bvd_55m | 50M | 4e-05 | 1.4e19 | 55.9 | 75.3 | 52.4 | 37.7 | 40.8 | 49.4 | 81.3 | 56.8 |
| ViCLIP-original baseline | - |  | - | - | - | 61.8 | 78.6 | 55.1 | 38.7 | 39.8 | 50.9 | 74.1 | 58.0 |
| ViCLIP_ViT-L-14 | 32000 | bvd_10m | 10M | 0.0001 | 1.3e19 | 62.7 | 79.5 | 60.4 | 42.7 | 42.9 | 53.2 | 81.0 | 61.3 |
| ViCLIP_ViT-L-14 | 32000 | bvd_10m | 50M | 4e-05 | 6.3e19 | 63.1 | 79.4 | 60.3 | 41.3 | 43.1 | 53.2 | 83.0 | 61.4 |
| ViCLIP_ViT-L-14 | 32000 | bvd_50m | 10M | 0.0001 | 1.3e19 | 63.3 | 78.9 | 60.4 | 43.5 | 42.6 | 53.6 | 82.1 | 61.5 |
| ViCLIP_ViT-L-14 | 32000 | bvd_55m | 10M | 0.0001 | 1.3e19 | 63.3 | 78.5 | 60.6 | 43.6 | 44.3 | 53.7 | 83.4 | 61.9 |
| ViCLIP_ViT-L-14 | 32000 | bvd_55m | 50M | 4e-05 | 6.3e19 | 63.3 | 80.5 | 60.4 | 42.7 | 43.0 | 53.6 | 84.3 | 62.0 |
| ViCLIP_ViT-L-14 | 32000 | bvd_50m | 50M | 4e-05 | 6.3e19 | 63.3 | 79.7 | 61.4 | 42.8 | 43.2 | 53.7 | 83.9 | 62.0 |




We also provide results with checkpoint merging:

```
pythonn summarize_wise_ft_results.py
```
