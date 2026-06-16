#!/bin/bash

OUTPUT_DIR="/m2v_intern/maoxinyu03/CineCap/TimeChat-Captioner/CineCap_Eval/0331_gated_cov_infer_cinecap472"


# 创建输出目录
mkdir -p "$OUTPUT_DIR"

MODEL_PATH="/m2v_intern/maoxinyu03/CineCap/TimeChat-Captioner/Train/output/0330_gemini_grpo_with_gated_cov/v0-20260330-085536/checkpoint-74"
FILE_NAME='0331_gated_cov'

# ==================== 启动循环 ====================
# 循环启动 8 个进程
for rank in {0..7}; do
    echo "[$(date '+%H:%M:%S')] Launching worker $rank on GPU $rank..."
    
    # nohup 后台运行，日志分开存
    nohup /root/miniconda3/envs/vllm/bin/python /m2v_intern/maoxinyu03/CineCap/TimeChat-Captioner/CineCap_Eval/cinecap_qwen3_infer.py \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        --file_name "$FILE_NAME" \
        --prompt "Describe the cinematic aspects in the video. \n Cinematic aspects include: Camera Movement, Depth of Field, Camera Angle, Subject Orientation, Shot Size, Composition, Special Shots. \n You should firstly watch the video and get visual evidence in the <think> block and then output the dense caption in the <answer> block." \
        --rank $rank \
        --world_size 8 \
        > "${OUTPUT_DIR}/worker_${rank}.log" 2>&1 &
    
    echo "Waiting 10s for worker $rank to initialize..."
    sleep 10
done

echo "✅ All 8 workers launched! Check logs in ${OUTPUT_DIR}/worker_*.log"