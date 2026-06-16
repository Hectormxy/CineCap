set -x

export DS_ENV_FILE="Train/config/deepspeed_env"

model="/your/sft/model/ckpt/here"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cinecap

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

sed -i "s/slots=1/slots=8/" /etc/mpi/hostfile
sed -i "s/slots=1/slots=8/" /etc/mpi/mpi-hostfile
hostfile="/etc/mpi/hostfile"

ts=`date +%Y%m%d-%H%M%S`
EXP_NAME="cinecap_grpo_with_gated_cov"
ACC_THRESHOLD=0.75
export CINECAP_ACC_THRESHOLD=${ACC_THRESHOLD}

DATASET="Data/grpo_data.json"
OUTPUT_DIR="Train/output/cinecap_grpo_with_gated_cov"
LOG_DIR="Train/log/${EXP_NAME}_${ts}"
mkdir -p "${LOG_DIR}"

GRPO_CLI_PATH="ThirdPartyLib/ms-swift/swift/cli/rlhf.py"

export GOOGLE_APPLICATION_CREDENTIALS="/you/should/add/your/application/credentials/here"

deepspeed --hostfile $hostfile \
        ${GRPO_CLI_PATH} \
        --rlhf_type grpo \
        --model $model \
        --reward_funcs external_cinecap_completeness_reward external_cinecap_accuracy_reward external_cinecap_aspect_coverage_reward\
        --reward_weights 0.5 0.5 0.1\
        --train_type full \
        --torch_dtype bfloat16 \
        --dataset ${DATASET} \
        --external_plugins Train/modified/cinecap_plugin.py  \
        --max_completion_length 9216 \
        --max_length 32768 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-5 \
        --freeze_vit true \
        --freeze_aligner false \
        --gradient_accumulation_steps 2 \
        --beta 0.04 \
        --save_steps 10 \
        --save_only_model true \
        --save_total_limit 5 \
        --logging_steps 1 \
        --remove_unused_columns false \
        --output_dir "${OUTPUT_DIR}" \
        --logging_dir "${LOG_DIR}" \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --num_generations 8 \
        --temperature 1. \
        --top_p 0.99 \
        --top_k 50 \
        --deepspeed Train/modified/zero3.json \
        --attn_impl flash_attn \
        --log_entropy \
        --log_completions true \
        --report_to tensorboard 2>&1 | grep --line-buffered -v "Unused or unrecognized kwargs" | grep --line-buffered -v -e "UserWarning: PySoundFile failed. Trying audioread instead." -e "FutureWarning: librosa.core.audio.__audioread_load" -e "Deprecated as of librosa version 0.10.0." -e "It will be removed in librosa version 1.0." -e "y, sr_native = __audioread_load(path, offset, duration, dtype)" -e "librosa.load(video, sr=self.sampling_rate)" | tee "${LOG_DIR}/${EXP_NAME}_${ts}.log"
