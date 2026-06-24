# CineCap: Structured Reasoning with Spatio-Temporal Anchors for Cinematographic Video Captioning

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2606.24636-b31b1b)](https://arxiv.org/abs/2606.24636)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/hector-mao/CineCap-GRPO-8B)
[![Benchmark](https://img.shields.io/badge/🤗%20Hugging%20Face-Benchmark-yellow)](https://huggingface.co/datasets/hector-mao/CineCap-Bench)

</div>

---

## 🌟 Overview

**CineCap** is a vision-language model for cinematographic captioning, which generates open-form descriptions of how a video is filmed across professional film-language dimensions such as camera movement, shot size, depth of field, composition, and shooting angle. Built on structured spatio-temporal reasoning and GRPO optimization, the model aims to produce captions that are both comprehensive and factually accurate for fine-grained cinematic video understanding.


- **🏠 Model:** [CineCap-GRPO-8B](https://huggingface.co/hector-mao/CineCap-GRPO-8B)
- **🏆 Benchmark:** [CineCap-Bench](https://huggingface.co/datasets/hector-mao/CineCap-Bench)


---

## 🚀 Quick Start

Below, we provide simple examples to show how to use CineCap-GRPO-8B with 🤗 Transformers.

### Installation

```bash
conda create -n cinecap python=3.12
conda activate cinecap
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install transformers==4.57.1
pip install qwen_vl_utils==0.0.14
pip install accelerate
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install deepspeed==0.16.9
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install "decord" -U
pip install msgspec
pip install -q -U google-genai
pip install func-timeout
pip install deepspeed==0.16.9
```

### Usage

> **Note:** To generate high-quality captions, limit video input to around 40 seconds. Please segment longer videos into around 40-second clips before processing.

```python
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# 1. Configuration
MODEL_ID = "hector-mao/CineCap-GRPO-8B"
VIDEO_PATH = "example_video.mp4"  # Replace with your video path

SYSTEM_PROMPT = "You are a video cinematography expert."

USER_PROMPT = (
    "Describe the cinematic aspects in the video.\n"
    "Cinematic aspects include: Camera Movement, Depth of Field, Camera Angle, "
    "Subject Orientation, Shot Size, Composition, Special Shots.\n"
    "You should firstly watch the video and get visual evidence in the <think> block "
    "and then output the dense caption in the <answer> block."
)


def extract_answer_text(text: str) -> str:
    """Extract the final caption from the <answer> block if present."""
    if not text:
        return ""

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()

    return re.sub(r"\s+", " ", text).strip()


print(f"🚀 Processing video: {VIDEO_PATH}")

# 2. Load model and processor
print("⏳ Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # remove this line if flash-attn is not installed
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# 3. Construct conversation
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "fps": 2.0,
                "max_frames": 80,
            },
            {
                "type": "text",
                "text": USER_PROMPT,
            },
        ],
    },
]

# 4. Process multimodal inputs
print("⚙️ Processing inputs...")

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    image_patch_size=processor.image_processor.patch_size,
    return_video_kwargs=True,
    return_video_metadata=True,
)

# Qwen3-VL returns video inputs as (video_tensor, video_metadata).
if video_inputs is not None:
    video_inputs, video_metadata = zip(*video_inputs)
    video_inputs = list(video_inputs)
    video_metadata = list(video_metadata)
else:
    video_metadata = None

inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    video_metadata=video_metadata,
    return_tensors="pt",
    do_resize=False,
    **video_kwargs,
)

inputs = inputs.to(model.device)

# 5. Generate cinematographic caption
print("✨ Generating caption...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
    )

# Remove input tokens from generated output.
generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

raw_output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

caption = extract_answer_text(raw_output)

print("\n" + "=" * 50)
print("🎬 CINEMATOGRAPHIC CAPTION:")
print("=" * 50)
print(caption)
print("=" * 50)
```

---

## 📊 Inference on CineCapBench

We provide a multi-GPU batch inference pipeline to evaluate CineCap on the [CineCap-Bench](https://huggingface.co/datasets/hector-mao/CineCap-Bench) benchmark.

**Step 1.** Download and extract the benchmark videos:

```bash
# Clone the dataset
git clone https://huggingface.co/datasets/hector-mao/CineCap-Bench CineCapBench

```

**Step 2.** Edit `Infer/infer.sh` to set your paths (`MODEL_PATH`, etc.).

**Step 3.** Run inference:

```bash
cd Infer
bash run_cinecap_infer.sh
```


---

## 🔧 Train

Training can be launched using the scripts provided in `Train/script/*.sh`.

---

## 📝 TODOs

- [ ] Upload readme for train and eval.

---

## 📖 Citation

```bibtex
@misc{mao2026cinecapstructuredreasoningspatiotemporal,
      title={CineCap: Structured Reasoning with Spatio-Temporal Anchors for Cinematographic Video Captioning}, 
      author={Xinyu Mao and Yuhui Zeng and Xiaokun Liu and Wenyu Qin and Meng Wang and Xin Tao and Pengfei Wan and Xiaohan Xing and Max Meng},
      year={2026},
      eprint={2606.24636},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.24636}, 
}
```


