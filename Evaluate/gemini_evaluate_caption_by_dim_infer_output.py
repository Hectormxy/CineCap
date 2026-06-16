#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate generated captions against GT captions using Gemini (Vertex AI).
The script streams responses from the Gemini API, extracts the JSON evaluation results, 
and writes them to the output file.
"""
# export GOOGLE_APPLICATION_CREDENTIALS="/m2v_intern/maoxinyu03/chatgpt-client/keling-ylab-gemini-1038ec8509a2.json"
import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
fmt_str = '%(asctime)s.%(msecs)03d %(levelname)7s [%(thread)d][%(process)d] %(message)s'
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
]

GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.,  # 保持极低的温度以保证评估标准的绝对稳定
    topP=1.0,
    topK=1,
    maxOutputTokens=65536,
    safety_settings=SAFETY_SETTINGS,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-grained Cinematic Caption Evaluation using Gemini (Vertex AI)."
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to the output JSON file.')
    
    parser.add_argument('--project', default="keling-ylab-gemini")
    parser.add_argument('--location', default="global",
                        help='Vertex AI location (default: global).')
    parser.add_argument('--model-name', default='gemini-3.1-pro-preview',
                        help='Gemini model name.')
    parser.add_argument('--max-workers', type=int, default=32, help='Thread pool size (default: 32).')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='Max retries per sample (default: 5).') 
    parser.add_argument('--retry-delay', type=float, default=5.0,
                        help='Seconds to wait between retries (default: 5.0).')
    return parser.parse_args()


def build_prompts(gt_caption: str, caption_a: str) -> str:
    """Return the structured evaluation prompt."""
    query = f"""
# Role
You are an expert Cinematic Caption Evaluator.

# Task
Compare "Candidate Caption A" against the "Ground Truth (GT) Caption". Break both captions into atomic claims under the 7 predefined cinematic aspects, match them contextually, and compute Comprehensiveness and Accuracy.

# Atomic Claim Rules
An atomic claim is one indivisible factual statement about ONE cinematic aspect.

1. **Action Splitting / Temporal Tracking**
Split sequential actions into separate claims.
- Example: "first pan right, then tilt up" -> ["pan right initially", "tilt up afterwards"]

2. **Semantic Deduplication**
Merge synonyms describing the same state at the same time.
- Example: "very high / from above / drone perspective" -> one claim: "high angle / drone perspective"

3. **Subject Co-reference**
When matching GT and Pred, resolve subject identity contextually.
- Example: "the man" and "the guitarist" are a match if they refer to the same entity.
- Do not penalize valid synonym, pronoun, or descriptive replacement.

Ignore claims outside the 7 aspects below.

# Evaluation Aspects
1. Camera Movement
   - Distinguish movement types strictly: zoom != move forward
   - Merge "handheld" + "shake" into "shake with a handheld device"
   - Ignore movement speed
2. Depth of Field
3. Camera Angle
4. Subject Orientation
   - Focus only on orientation
   - "front and side" means profile
5. Shot Size
   - Distinguish strictly: close-up, medium close-up, medium, full, wide/long/panorama
6. Composition
   - Subject position only: center, left, right, center-left, center-right
7. Special Shots
   - Only if explicitly present: POV, fisheye, slow motion, fast motion, over the shoulder, timelapse, bullet time

# Input
- GT: "{gt_caption}"
- A: "{caption_a}"

# Thinking Steps
Use a <think> block.

Step 1. Extract GT atomic claims and group them by aspect.

Step 2. Extract A atomic claims and group them by aspect.

Step 3. For each aspect, compare GT claims against A claims:
- [Matched] if logically covered
- [Missing] otherwise

Step 4. For each aspect, compare A claims against GT claims:
- [Correct] if supported by GT
- [Error] if contradicted or hallucinated

Step 5. For each aspect, output:
- gt_statements
- pred_statements
- matched_statements
- gt_count
- pred_count
- matched_count

Then aggregate:
- Total GT Claims = sum(gt_count)
- Total Pred Claims = sum(pred_count)
- Total Matched Claims = sum(matched_count)
- Comprehensiveness = Total Matched Claims / Total GT Claims
- Accuracy = Total Matched Claims / Total Pred Claims

Notes:
- If an aspect has no GT claim: gt_statements=[], gt_count=0
- If an aspect has no Pred claim: pred_statements=[], pred_count=0
- If an aspect has no match: matched_statements=[], matched_count=0
- If Total GT Claims = 0, Comprehensiveness = 1.0
- If Total Pred Claims = 0, Accuracy = 0.0
- matched_statements must be GT-side statements that are correctly covered by A
- Output only the 7 predefined aspects

# Output
After </think>, output ONLY a valid JSON object inside <answer>.

<answer>
{{
  "by_dim": {{
    "Camera Movement": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Depth of Field": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Camera Angle": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Subject Orientation": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Shot Size": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Composition": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }},
    "Special Shots": {{
      "gt_statements": [],
      "pred_statements": [],
      "matched_statements": [],
      "gt_count": 0,
      "pred_count": 0,
      "matched_count": 0
    }}
  }},
  "gt_claims_count": 0,
  "matched_claims_count": 0,
  "missing_claims_count": 0,
  "comprehensiveness_score": 0.00,
  "a_claims_count": 0,
  "error_claims_count": 0,
  "accuracy_score": 0.00
}}
</answer>
""".strip()
    return query


def generate_text(
    client: genai.Client,
    model_name: str,
    contents: List[types.Content],
) -> str:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GENERATION_CONFIG,
        )
    except Exception as e:
        raise RuntimeError(f"API Connection Error: {e}")

    if not response.candidates:
        raise ValueError(f"No candidates returned. Feedback: {response.prompt_feedback}")

    candidate = response.candidates[0]

    has_content = candidate.content and candidate.content.parts and candidate.content.parts[0].text
    if not has_content:
        logger.error("!!! DETECTED EMPTY CONTENT !!!")
        logger.error(f"Finish Reason: {candidate.finish_reason}")
        if candidate.safety_ratings:
            logger.error("Safety Ratings Details:")
            for rating in candidate.safety_ratings:
                logger.error(f"  - {rating.category}: {rating.probability} (Blocked: {rating.blocked})")
        raise ValueError(f"Content is empty. Reason: {candidate.finish_reason}")

    return candidate.content.parts[0].text.strip()


def run_one_turn_chat(
    project: str,
    location: str,
    model_name: str,
    prompt: str,
) -> str:
    """Execute Gemini call."""
    client = genai.Client(vertexai=True, project=project, location=location)

    user_turn = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )
    answer = generate_text(client, model_name, [user_turn])
    return answer


def process_single_item(
    item: Dict[str, Any],
    project: str,
    location: str,
    model_name: str,
    max_retries: int,
    retry_delay: float,
) -> Optional[Dict[str, Any]]:
    """Apply the Gemini evaluation workflow to a single sample."""
    # 提取新的数据格式
    gt_caption = item.get("gt_caption")
    # caption_a = item.get("pred")
    caption_a = item.get("predictions")
    # caption_a = item.get("response")
    video_path = item.get("video_path")
    
    if not all([gt_caption, caption_a, video_path]):
        logger.error("Missing required fields in item: %s", item)
        return None

    prompt = build_prompts(gt_caption, caption_a)
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            answer = run_one_turn_chat(project, location, model_name, prompt)
            
            # 使用正则精准提取 <answer> 块内的 JSON
            # 考虑模型偶尔可能省略 <answer> 标签的情况，直接抓取首尾 {}
            json_match = re.search(r'\{.*\}', answer, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                checklist_data = json.loads(json_string)

                com_score = checklist_data.get("comprehensiveness_score", 0.0)
                acc_score = checklist_data.get("accuracy_score", 0.0)

                return {
                    "video_path": video_path,
                    "success": 1,
                    "gt_caption": gt_caption,
                    "A": caption_a,
                    "com": com_score,    # 提取的 Comprehensiveness 分数
                    "acc": acc_score,    # 提取的 Accuracy 分数
                    "checklist": checklist_data,
                    "response": answer
                }
            else:
                raise ValueError("No JSON object found in the model response.")
                
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "Retry %d/%d failed for %s: %s",
                attempt + 1,
                max_retries,
                video_path,
                exc,
            )
            time.sleep(retry_delay)

    logger.error(
        "Skip %s after %d retries due to error: %s",
        video_path,
        max_retries,
        last_exception,
    )
    return None


def main() -> None:
    args = parse_args()
    if not args.project:
        raise SystemExit("GCP project must be set via --project or GCP_PROJECT env.")

    # 1. 加载输入数据
    logger.info(f"Loading input data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    logger.info("Total samples to process: %d", len(input_data))

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    
    # 动态生成 failure 文件的路径
    results_path = args.output
    failures_path = args.output.replace('.json', '_failures.json')

    # 2. 线程池并发处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {}
        for idx, item in enumerate(input_data):
            future = executor.submit(
                process_single_item,
                item,
                args.project,
                args.location,
                args.model_name,
                args.max_retries,
                args.retry_delay,
            )
            future_to_idx[future] = idx

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is None:
                    failures.append(input_data[idx])
                    continue
                results.append(result)
            except Exception as exc:
                logger.error("Error processing index %d: %s", idx, exc)
                failures.append(input_data[idx])

            # 实时保存结果
            with open(results_path, 'w', encoding='utf-8') as outfile:
                json.dump(results, outfile, ensure_ascii=False, indent=4)
            if failures:
                with open(failures_path, 'w', encoding='utf-8') as failfile:
                    json.dump(failures, failfile, ensure_ascii=False, indent=4)

    logger.info("Completed. %d successes, %d failures.", len(results), len(failures))
    logger.info("Results saved to: %s", results_path)


if __name__ == '__main__':
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("tqdm is required: pip install tqdm") from exc

    main()