#!/bin/bash -x
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={train_time}
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
export LD_LIBRARY_PATH=/path/to/mamba/envs/torch-2.10.0/lib:$LD_LIBRARY_PATH # for ffmpeg / torchcodec
ulimit -c 0
srun python -u /path/to/open_clip_video/src/open_clip_train/main.py \
    --save-frequency 1 \
    --pretrained {pretrained} \
    --dataset-type webdataset_video \
    --train-data "{train_data}" \
    --train-num-samples "{train_num_samples}" \
    --warmup {warmup} \
    --batch-size {batch_size} \
    --report-to=tensorboard \
    --epochs={epochs} {dataset_args} \
    --workers={workers} \
    --model {model} \
    --name {experiment_name} \
    --logs {logs} \
    --seed {seed} \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --lr {lr} {lock}\
    --save-most-recent \
    --precision amp_bfloat16 \
    --grad-checkpointing \
    --grad-clip-norm {grad_clip_norm} \
    --wd {wd} \
    --beta1 {beta1} \
    --beta2 {beta2} \
    --resume latest \
    --wds-caption-key "txt;caption.txt" \
    --cache-dir $OPEN_CLIP_CACHE_DIR \
    --video-decoder-num-ffmpeg-threads 16 \
    --log-every-n-steps 1
