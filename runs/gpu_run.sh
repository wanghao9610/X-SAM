#!/usr/bin/env bash
# set -x

#######################################################################
#                          PART 1  Environment                         #
#######################################################################
## Logformat
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

## CUDA
echo -e "$log_format Nvidia-smi : \n$(nvidia-smi)"
echo -e "$log_format Cuda : \n$(nvcc -V)"
echo -e "$log_format Conda : \n$(which python)"

#######################################################################
#                          PART 2  Project Config                      #
#######################################################################
## Directory
PROJ_HOME=${PROJ_HOME:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}
code_name="xsam"
code_dir="$PROJ_HOME/$code_name"
data_dir="$PROJ_HOME/datas"
init_dir="$PROJ_HOME/inits"
work_dir="$PROJ_HOME/wkdrs"
exec_dir=$HJOB_DIR/exec
hostfile=$HJOB_DIR/hostfile
export ROOT_DIR="$PROJ_HOME/"
export DATA_DIR="$data_dir/"
export INIT_DIR="$init_dir/"
export WORK_DIR="$work_dir/"

export LMUData="$data_dir/LMUData"
export HF_HOME="$init_dir/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_DATASET_TIMEOUT=120
export USE_TRITON_KERNEL=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0

## Debug
# export DEBUG_MODE=true
# export DEBUG_ITEM=256
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

## Model Config
deepspeed_config="xsam/configs/deepspeed/deepspeed_zero2.json"
config_file="${1:-xsam/xsam/configs/xsam/s3_train/xsam_qwen3_vl_2b_instruct_sam_vit_large_m2f_e1_gpu16_lora.py}"
read -r -a modes <<< "${2:-train segeval vlmeval visualize}"
shift 2

# Optional: [prefix] [suffix] [-- extra...]; skip slot with "-", e.g. ["-" "debug"] or ["prefix" "suffix"] or leave empty
prefix=""; suffix=""
for n in 1 2; do
    [[ $# -eq 0 || "$1" == -- || ( "$1" == -* && "$1" != "-" ) ]] && break
    [[ "$1" == "-" ]] && { shift; continue; }
    [[ "$n" == 1 ]] && prefix="$1" || suffix="${1#_}"
    shift
done
[[ "${1:-}" == -- ]] && shift

extra=("$@")
stage="$(basename "$(dirname "$config_file")")"
model_name="$(basename "$config_file" .py)"
vlm_name="$(sed -E 's/_e[0-9]+_gpu[0-9]+//; s/_/-/g' <<<"$model_name")"
# work_dir="$work_dir/$stage/${prefix:+$prefix/}$model_name${suffix:+_$suffix}"
work_dir="$work_dir/gpu_debug"
lscp_file="$work_dir/last_checkpoint"

## Distributed Training Config
num_nodes=${NUM_NODES:-1}
node_rank=${NODE_RANK:-0}
gpu_per_node=${GPU_PER_NODE:-$(nvidia-smi -L | wc -l)}
master_addr=${MASTER_ADDR:-"127.0.0.1"}
master_port=${MASTER_PORT:-29510}

rm -rf "$PROJ_HOME/wkdrs/gpu_debug"

# #######################################################################
#                          PART 3  Run Config                          #
# #######################################################################

## Retry Config (for node failure recovery)
## max_retries: max number of retries before giving up
## retry_wait:  seconds to wait before retrying (allow crashed node to restart)
## retry_node_delay: extra seconds non-rank-0 nodes sleep per retry (let rank-0 reconnect first)
max_retries=100
retry_wait=30
retry_node_delay=30

## run_with_retry <log_file> <cmd> [args...]
##   Retries <cmd> up to $max_retries times on failure.
##   Captures torchrun exit code correctly from the left side of the pipe via PIPESTATUS.
##   Appends each attempt's output to <log_file> (rank-0 only); pass "" to skip logging.
run_with_retry() {
    local log_file="$1"; shift
    local attempt=1
    while true; do
        log_time=$(date "+%Y-%m-%d %H:%M:%S")
        log_format="[$log_time] [INFO] [$0]"
        echo -e "$log_format [Attempt $attempt/$max_retries] Starting."
        if [ -n "$log_file" ] && [ "$node_rank" -eq 0 ]; then
            "$@" | tee -a "$log_file"
            exit_code=${PIPESTATUS[0]}
        else
            "$@"
            exit_code=$?
        fi
        if [ $exit_code -eq 0 ]; then
            return 0
        fi
        if [ $attempt -ge $max_retries ]; then
            log_time=$(date "+%Y-%m-%d %H:%M:%S")
            log_format="[$log_time] [INFO] [$0]"
            echo -e "$log_format Command failed after $max_retries attempts (exit=$exit_code). Giving up." >&2
            return $exit_code
        fi
        log_time=$(date "+%Y-%m-%d %H:%M:%S")
        log_format="[$log_time] [INFO] [$0]"
        echo -e "$log_format Attempt $attempt/$max_retries failed (exit=$exit_code). Waiting ${retry_wait}s for node recovery..." >&2
        sleep $retry_wait
        # Stagger non-rank-0 nodes so rank-0 reconnects to the rendezvous first
        [ "$node_rank" -ne 0 ] && sleep $retry_node_delay
        attempt=$((attempt + 1))
    done
}

## Run
for mode in "${modes[@]}"
do  
    echo -e "$log_format Mode: $mode."
    time=$(date "+%Y%m%d-%H%M%S")
    [ "$node_rank" -ne 0 ] && sleep 30
    if [ "$mode" = "train" ] && [ ! -d "$work_dir" ] && [ "$node_rank" = 0 ]; then
        mkdir -p "$work_dir"
        cp -rf "$(realpath "$0")" "$code_dir" "$work_dir"
        find "$work_dir/$code_name" -name "*.crc" -type f -delete
        find "$work_dir/$code_name" -type f -exec chmod 664 {} +
        find "$work_dir/$code_name" -type d -exec chmod 775 {} +
    fi
    if [ -d "$work_dir" ]; then
        cp "$(realpath "$0")" "$work_dir"
        if [ -d "$work_dir/$code_name" ]; then
            code_dir="$work_dir/$code_name"
        fi
    fi
    cd "$code_dir"
    export CODE_DIR="$code_dir/"
    echo -e "$log_format code_dir: $code_dir"

    [ -f "$config_file" ] || config_file="${config_file#$code_name/}"
    [ -f "$config_file" ] || { echo -e "$log_format Config file not found: $config_file" >&2; exit 1; }
    
    ## mode: train
    trained_flag=0
    if [ "$mode" = "train" ]; then
        run_with_retry "$work_dir/train-${time}.log" \
            env PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
            "$code_dir/xsam/tools/train.py" \
            "$config_file" \
            --work-dir "$work_dir" \
            --resume auto \
            --launcher pytorch \
            --deepspeed "$deepspeed_config" \
            --seed 1024 \
            "${extra[@]}"
    fi
    if [ -f "$lscp_file" ] || [[ ! " ${modes[*]} " =~ " train " ]]; then
        trained_flag=1
    fi
    
    ## mode: segeval
    if [ "$mode" = "segeval" ]; then
        if [ "$node_rank" = 0 ] && [ -f "$work_dir/last_checkpoint" ]; then
            echo -e "$log_format Converting $model_name to PT format."
            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH \
                python "$code_dir/xsam/tools/model_tools/ds_zero_to_pytorch.py" \
                --config \
                "$config_file" \
                --work-dir \
                "$work_dir" \
                "${extra[@]}"
        fi
        run_with_retry "$work_dir/segeval-${time}.log" \
            env PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
            "$code_dir/xsam/tools/eval.py" \
            "$config_file" \
            --launcher pytorch \
            --work-dir "$work_dir" \
            --pth_model latest \
            --seed 0 \
            "${extra[@]}"
    fi

    ## mode: vlmeval
    if [ "$mode" = "vlmeval" ] && [ "$trained_flag" = 1 ]; then
        if [ "$node_rank" = 0 ] && [ ! -d "$work_dir/pytorch_model" ]; then
            echo -e "$log_format Converting $model_name to HF format."
            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH \
                python "$code_dir/xsam/tools/model_tools/pth_to_hf.py" \
                "$code_dir/$config_file" \
                "$work_dir" \
                --pth_model latest \
                --save-format huggingface
            rm "$init_dir/$vlm_name"
            ln -s "$work_dir/pytorch_model" "$init_dir/$vlm_name"
        fi
        if [ -d "$work_dir/pytorch_model" ]; then
            echo -e "$log_format Evaluating VLM: $model_name."
            [ "$node_rank" -ne 0 ] && sleep 30
            run_with_retry "$work_dir/vlmeval-${time}.log" \
                env PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
                "$code_dir/xsam/evaluation/vlmeval/run.py" \
                --data MME MMBench_DEV_EN SEEDBench_IMG POPE GQA_TestDev_Balanced AI2D_TEST ScienceQA_VAL \
                --model "$vlm_name" \
                --work-dir "$work_dir/vlmeval_results" \
                "${extra[@]}"
        fi
    fi

    ## mode: visualize
    if [ "$mode" = "visualize" ]; then
        echo -e "$log_format Visualizing $model_name."
        PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python "$code_dir/xsam/tools/visualize.py" \
            "$config_file" \
            --work-dir "$work_dir" \
            --pth_model latest \
            --concat-aux-img \
            --max-samples 200 \
            --seed 0 \
            "${extra[@]}" | { [ "$node_rank" = "0" ] && tee "$work_dir/visualize-${time}.log" || cat; }
    fi

    rm -rf /tmp/dataset_cache
done
