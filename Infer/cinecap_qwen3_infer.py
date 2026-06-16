import os
import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ==================== 环境设置 ====================
# 强制使用 V0 引擎避免初始化崩溃
os.environ["VLLM_USE_V1"] = "0"
# 关闭 Ray，减少干扰
os.environ["VLLM_WORKER_USE_RAY"] = "0"

def extract_answer_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r'\s+', ' ', match.group(1)).strip()
    return re.sub(r'\s+', ' ', text).strip()


def main():
    parser = argparse.ArgumentParser(description="Single GPU Inference")
    parser.add_argument('--model_path', type=str, default="/m2v_intern/maoxinyu03/code/ms-swift/caption_expert/exp/0209_swift_cine_reflection_full_sft_Qwen3_8b_fps2_bs64_maxfrm80_maxtoken256_bspd1_data8w_stage3_first/v6-20260210-064553/checkpoint-649")
    parser.add_argument('--file_name', type=str, default='0209_self_correction_infer')
    parser.add_argument('--data_path', type=str, default="/m2v_intern/maoxinyu03/benchmark/CineCap_full_test_benchmark_total_472.json")
    parser.add_argument('--output_dir', type=str, required=True)
    
    # [新增] 支持通过命令行传入 Prompt
    parser.add_argument('--prompt', type=str, 
                        default="Describe the cinematic aspects in the video. \n Cinematic aspects include: Camera Movement, Depth of Field, Camera Angle, Subject Orientation, Shot Size, Composition, Special Shots. \n First, generate an initial <draft> based on your first impression of the video. Then, enter a <think> block to verify and critique your draft. Finally, output the corrected dense caption in the <answer> block.",
                        help="The question template/prompt for the model.")
    
    parser.add_argument('--rank', type=int, required=True, help='Current GPU ID (0-7)')
    parser.add_argument('--world_size', type=int, default=8, help='Total GPUs')
    
    args = parser.parse_args()

    # 1. 物理隔离：只让当前进程看到这一张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rank)
    print(f"Worker {args.rank}: Starting on GPU {args.rank}...")

    # 2. 计算分片
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"Worker {args.rank}: Failed to load data: {e}")
        return

    total_samples = len(data)
    samples_per_process = (total_samples + args.world_size - 1) // args.world_size
    start_idx = args.rank * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_samples)
    
    process_data = data[start_idx:end_idx]
    print(f"Worker {args.rank}: Handling samples {start_idx} to {end_idx-1} ({len(process_data)} total)")

    # 3. 初始化模型
    try:
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1, 
            max_model_len=32768,  
            gpu_memory_utilization=0.60, 
            limit_mm_per_prompt={"image": 1, "video": 1},
            mm_encoder_tp_mode="data", 
            enable_expert_parallel=False, 
            trust_remote_code=True,
        )
        
        sampling_params = SamplingParams(
            n=1,                
            temperature=0.0,    
            top_p=1.0,
            max_tokens=4096,
            stop_token_ids=[],
        )
        
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Worker {args.rank}: Model init failed: {e}")
        return

    SYSTEM_PROMPT = "You are a video cinematography expert."

    # 4. 数据预处理
    processed_messages = []
    valid_data = [] # 记录有效数据，防止索引错位

    for item in process_data:
        video_path = item.get("video_path")
        
        if not video_path:
            print(f"Worker {args.rank}: Warning - Missing video_path in item.")
            continue

        # [修改] 使用 args.prompt 替代硬编码的 QUESTION_TEMPLATE
        text_content = args.prompt

        new_msgs = [
            # System Message
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            # User Message (多模态输入)
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": video_path, 
                        "fps": 2.0,        
                        "max_frames": 80   
                    },
                    {
                        "type": "text", 
                        "text": text_content
                    }
                ]
            }
        ]
        
        processed_messages.append(new_msgs)
        valid_data.append(item) 

    results = []
    output_file = os.path.join(args.output_dir, f"{args.file_name}_rank{args.rank}_sampled.json")
    
    # 断点续传逻辑
    resume_idx = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                resume_idx = len(results)
                print(f"Worker {args.rank}: Resuming from {resume_idx}")
        except:
            print(f"Worker {args.rank}: Output file corrupted, restarting.")

    # 开始推理
    for i in tqdm(range(resume_idx, len(processed_messages)), desc=f"GPU {args.rank}"):
        msg = processed_messages[i]
        original_item = valid_data[i] 

        gt_caption = original_item.get("gt_caption", "")

        try:
            text_prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                msg,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            mm_data = {}
            if image_inputs is not None: mm_data['image'] = image_inputs
            if video_inputs is not None: mm_data['video'] = video_inputs
            
            llm_input = {
                "prompt": text_prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs
            }
            
            outputs = llm.generate([llm_input], sampling_params=sampling_params)
            generated_texts = [o.text.strip() for o in outputs[0].outputs]
            pred_text = extract_answer_text(generated_texts[0]) if generated_texts else ""
            
            results.append({
                "video_path": original_item.get("video_path", ''),
                "gt_caption": gt_caption,
                "pred": pred_text,
                "predictions": generated_texts
            })
            
        except Exception as e:
            print(f"Worker {args.rank} Error on item {i}: {e}")
            results.append({
                "video_path": original_item.get("video_path", ''),
                "gt_caption": gt_caption,
                "pred": "",
                "error": str(e),
                "predictions": []
            })
        
        # 每10条保存一次
        if (i - resume_idx + 1) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Worker {args.rank}: Done.")

if __name__ == "__main__":
    main()