#!/bin/bash -x
#SBATCH --nodes={eval_nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={eval_time}
#SBATCH --output={output_file}
#SBATCH --error={output_file}
set -e
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export HF_HOME=/path/to/huggingface/cache
export HF_HUB_OFFLINE=1
export OPEN_CLIP_CACHE_DIR=/path/to/open_clip/cache
export WANDB_MODE="offline"
export TRITON_DUMP_DIR="/path/to/triton/dump"
mkdir -p $TRITON_DUMP_DIR
export TRITON_CACHE_DIR="/path/to/triton/cache"
mkdir -p $TRITON_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
OPEN_CLIP_PATH="/path/to/open_clip_video"
export PYTHONPATH="$OPEN_CLIP_PATH/src:$PYTHONPATH"
export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
source /path/to/mamba/bin/activate torch-2.10.0
export PYTHONPATH=/path/to/CLIP_benchmark_video:/path/to/open_clip_video/src:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/mamba/envs/torch-2.10.0/lib:$LD_LIBRARY_PATH # for ffmpeg / torchcodec
ulimit -c 0
# nevals: {nevals}
srun python -u `which clip_benchmark`  eval --model {model} --pretrained {logs}/{folder_name}/checkpoints/epoch_[0-9]*.pt --dataset {evals} --dataset_root "/path/to/video_benchmarks/{{dataset}}" --output '{logs}/{folder_name}/checkpoints/{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}.json' --skip_existing --distributed --recall_k 1 5 10 --num_workers 8 --model_cache_dir $OPEN_CLIP_CACHE_DIR
