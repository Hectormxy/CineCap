#!/bin/bash
export PATH=/usr/local/bin:$PATH
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")  # 自动找TCP网卡
LD_PRELOAD=$(cat /etc/xray_kccl_path 2>/dev/null || echo ""):$LD_PRELOAD

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


#配置多机文件的hostfile 多机时候确保每机读8卡
sed -i "s/slots=1/slots=8/" /etc/mpi/hostfile
sed -i "s/slots=1/slots=8/" /etc/mpi/mpi-hostfile
hostfile="/etc/mpi/hostfile"

ts=`date +%Y_%m_%d_%H_%M`
mkdir -p "logs"
RUN_NAME=0312_swift_cine_concise_cot_full_sft_Qwen3_8b_fps2_bs128_maxfrm80_maxtoken256_bspd2_data8w_stage3_first

export OMPI_MCA_btl=self,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_tcp_if_include=$TCP_NIC
export OMPI_MCA_oob_tcp_if_include=$TCP_NIC
export OMPI_MCA_btl_openib_allow_ib=false
export NCCL_IB_HCA=mlx5

export NCCL_IB_GDR_LEVEL=0


# 其他稳健参数
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=$TCP_NIC
export NCCL_IB_TC=128           # 加上 TC 配置，防止 RoCE 拥塞
export NCCL_IB_RETRY_CNT=15     # 多重试
export NCCL_IB_TIMEOUT=22       # 超时时间拉长
export NCCL_IB_QPS_PER_CONNECTION=4

# 加载 KCCL 库
export LD_PRELOAD=$LD_PRELOAD

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=2
NUM_DEVICES=32
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


python -m deepspeed.launcher.runner --hostfile ${hostfile} \
    /m2v_intern/maoxinyu03/code/ms-swift/swift/cli/sft.py \
    --model '/m2v_intern/maoxinyu03/Models/Qwen3-VL-8B-Instruct' \
    --dataset "/m2v_intern/maoxinyu03/CineCap/data/ms_swift_qwen3_gemini_concise_cot_8w.json" \
    --train_type full \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False \
    --torch_dtype bfloat16 \
    --model_kwargs '{"fps_max_frames": 80, "fps_min_frames": 4, "fps": 2, "video_min_token_num": 64, "video_max_token_num": 256}' \
    --gradient_checkpointing True \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --remove_unused_columns False \
    --learning_rate 2e-5 \
    --vit_lr 1e-6 \
    --aligner_lr 3e-5 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --save_steps 350 \
    --logging_steps 1 \
    --attn_impl 'flash_attention_2' \
    --output_dir /m2v_intern/maoxinyu03/code/ms-swift/caption_expert/exp/${RUN_NAME} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed /m2v_intern/maoxinyu03/code/ms-swift/caption_expert/ds_z2_config.json \
    --save_total_limit 2 2>&1 | tee ./logs/log_${RUN_NAME}_${ts}.log



# bash /m2v_intern/liuxiaokun/for_maoyuxin/gpu_storer/start.sh
