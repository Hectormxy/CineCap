import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from func_timeout import FunctionTimedOut, func_timeout
from google import genai
from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
from transformers import AutoTokenizer

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

TOKENIZER_PATH = '/m2v_intern/maoxinyu03/code/ms-swift/caption_expert/exp/0312_swift_cine_concise_cot_full_sft_Qwen3_8b_fps2_bs128_maxfrm80_maxtoken256_bspd2_data8w_stage3_first/v0-20260312-081814/checkpoint-1294'
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    return _TOKENIZER


class CineCapGeminiJudge:
    DIM_KEYS = [
        'Camera Movement',
        'Shot Size',
        'Depth of Field',
        'Camera Angle',
        'Composition',
        'Subject Orientation',
    ]

    def __init__(
        self,
        max_workers: int = 8,
        log_file: str = 'cinecap_gemini_judge.log',
        cache_dir: str = 'cache_cinecap_judge',
        model_name: str = 'gemini-2.5-flash',
        credentials: str = "/m2v_intern/maoxinyu03/chatgpt-client/keling-ylab-gemini-1038ec8509a2.json",
        eval_model_name: Optional[str] = None,
        debug: bool = False,
    ):
        self.max_workers = max_workers
        self.model_name = model_name
        self.credentials = credentials
        self.debug = debug
        self.eval_model_name = eval_model_name or time.strftime('run-%Y%m%d-%H%M')
        self.cache_dir = Path(cache_dir) / self.eval_model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials
        user_info = json.load(open(self.credentials))
        self.client = genai.Client(vertexai=True, project=user_info['project_id'], location='global')
        self.config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.001,
            safety_settings=[
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
            ],
            seed=42,
        )

    def _norm_text(self, x: Any) -> str:
        x = '' if x is None else str(x)
        return re.sub(r'\s+', ' ', x).strip()

    def _hash_key(self, gt_text: str, pred_text: str) -> str:
        raw = '||'.join([self.model_name, self._norm_text(gt_text), self._norm_text(pred_text)])
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f'{key}.json'

    def _safe_json(self, text: str) -> Dict[str, Any]:
        s = (text or '').strip()
        if s.startswith('```'):
            s = re.sub(r'^```[a-zA-Z]*\n?', '', s)
            s = re.sub(r'\n?```$', '', s).strip()
        first, last = s.find('{'), s.rfind('}')
        if first != -1 and last != -1 and last > first:
            s = s[first:last + 1]
        try:
            return json.loads(s)
        except Exception:
            return {}

    def _build_prompt(self, gt_text: str, pred_text: str) -> str:
        def fmt_dim(dim: str) -> str:
            return json.dumps(dim, ensure_ascii=False)

        return f"""
            You are a strict evaluator for camera-language captions.

            You will receive:
            (1) one ground-truth caption
            (2) one predicted caption

            Your task is to decompose both captions into atomic statements under exactly 6 aspects:
            - Camera Movement
            - Shot Size
            - Depth of Field
            - Camera Angle
            - Composition
            - Subject Orientation

            Rules:
            - Only use these 6 aspects. Do not create new aspects.
            - Each aspect may contain zero, one, or multiple atomic statements.
            - Statements must be atomic, explicit, and text-grounded.
            - Merge paraphrases with the same meaning.
            - Do not infer missing information.
            - A matched statement means the GT statement and Pred statement express the same semantic fact.
            - For each aspect, count how many GT statements are correctly covered by Pred.
            - For each aspect, also report GT statement count and Pred statement count.
            - Output STRICT JSON ONLY.
            - Do not output any explanation.
            - Do not output markdown.
            - Do not output code fences.
            - Do not output any text before or after the JSON.
            - If an aspect has no valid statements, use empty lists and count 0.

            Output format (STRICT JSON ONLY):
            {{
              "by_dim": {{
                "Camera Movement": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }},
                "Shot Size": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }},
                "Depth of Field": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }},
                "Camera Angle": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }},
                "Composition": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }},
                "Subject Orientation": {{
                  "gt_statements": [<string>, ...],
                  "pred_statements": [<string>, ...],
                  "matched_statements": [<string>, ...],
                  "gt_count": <int>,
                  "pred_count": <int>,
                  "matched_count": <int>
                }}
              }},
              "total_gt_count": <int>,
              "total_pred_count": <int>,
              "total_matched_count": <int>
            }}

            The 6 aspects are exactly:
            - {fmt_dim('Camera Movement')}
            - {fmt_dim('Shot Size')}
            - {fmt_dim('Depth of Field')}
            - {fmt_dim('Camera Angle')}
            - {fmt_dim('Composition')}
            - {fmt_dim('Subject Orientation')}

            Ground-truth caption:
            {self._norm_text(gt_text)}

            Predicted caption:
            {self._norm_text(pred_text)}

            Return JSON only.
            """.strip()

    def _call_once(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.models.generate_content(model=self.model_name, contents=[prompt], config=self.config)
        text = ''.join(p.text for p in resp.candidates[0].content.parts if hasattr(p, 'text')).strip()
        return self._safe_json(text)

    def _empty_result(self) -> Dict[str, Any]:
        by_dim = {}
        for dim in self.DIM_KEYS:
            by_dim[dim] = {
                'gt_statements': [],
                'pred_statements': [],
                'matched_statements': [],
                'gt_count': 0,
                'pred_count': 0,
                'matched_count': 0,
            }
        return {
            'by_dim': by_dim,
            'total_gt_count': 0,
            'total_pred_count': 0,
            'total_matched_count': 0,
        }

    def _call_retry(self, prompt: str, retries: int = 4, base_delay: float = 1.2, timeout_sec: int = 180) -> Dict[str, Any]:
        for attempt in range(retries):
            wait = base_delay * (2 ** attempt)
            try:
                result = func_timeout(timeout_sec, self._call_once, args=(prompt,))
                if isinstance(result, dict) and 'by_dim' in result:
                    return result
                raise ValueError(f'invalid judge result: {str(result)[:120]}')
            except FunctionTimedOut:
                print(f'[Gemini Judge] attempt={attempt + 1}/{retries} timeout (> {timeout_sec}s)')
            except Exception as e:
                print(f'[Gemini Judge] attempt={attempt + 1}/{retries} failed: {e}; sleep {wait:.1f}s')
            time.sleep(wait)
        return self._empty_result()

    def _sanitize_count(self, value: Any) -> int:
        try:
            count = int(value)
        except Exception:
            return 0
        return max(0, count)

    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        safe = self._empty_result()
        by_dim = result.get('by_dim', {}) if isinstance(result, dict) else {}
        total_gt = 0
        total_pred = 0
        total_matched = 0

        for dim in self.DIM_KEYS:
            dim_value = by_dim.get(dim, {}) if isinstance(by_dim, dict) else {}
            gt_statements = dim_value.get('gt_statements', [])
            pred_statements = dim_value.get('pred_statements', [])
            matched_statements = dim_value.get('matched_statements', [])
            if not isinstance(gt_statements, list):
                gt_statements = []
            if not isinstance(pred_statements, list):
                pred_statements = []
            if not isinstance(matched_statements, list):
                matched_statements = []

            gt_count = self._sanitize_count(dim_value.get('gt_count', len(gt_statements)))
            pred_count = self._sanitize_count(dim_value.get('pred_count', len(pred_statements)))
            matched_count = self._sanitize_count(dim_value.get('matched_count', len(matched_statements)))
            matched_count = min(matched_count, gt_count, pred_count)

            safe['by_dim'][dim] = {
                'gt_statements': [self._norm_text(x) for x in gt_statements if self._norm_text(x)],
                'pred_statements': [self._norm_text(x) for x in pred_statements if self._norm_text(x)],
                'matched_statements': [self._norm_text(x) for x in matched_statements if self._norm_text(x)],
                'gt_count': gt_count,
                'pred_count': pred_count,
                'matched_count': matched_count,
            }
            total_gt += gt_count
            total_pred += pred_count
            total_matched += matched_count

        safe['total_gt_count'] = total_gt
        safe['total_pred_count'] = total_pred
        safe['total_matched_count'] = min(total_matched, total_gt, total_pred)
        return safe

    def score_pair(self, gt_text: str, pred_text: str) -> Dict[str, Any]:
        gt_text = self._norm_text(gt_text)
        pred_text = self._norm_text(pred_text)
        key = self._hash_key(gt_text, pred_text)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                return self._sanitize_result(json.load(open(cache_path)))
            except Exception:
                pass

        prompt = self._build_prompt(gt_text, pred_text)
        result = self._sanitize_result(self._call_retry(prompt))

        with self._write_lock:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f'write cache failed: {e}')
        return result


def _normalize_solution_item(solution_item: Any) -> str:
    if solution_item is None:
        return ''
    if isinstance(solution_item, str):
        return solution_item.strip()
    if isinstance(solution_item, dict):
        for key in ['caption', 'answer', 'response', 'content', 'solution', 'gt']:
            value = solution_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = []
        for _, value in solution_item.items():
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return '\n'.join(parts).strip()
    if isinstance(solution_item, list):
        parts = [_normalize_solution_item(item) for item in solution_item]
        parts = [item for item in parts if item]
        return '\n'.join(parts).strip()
    return str(solution_item).strip()


def _extract_answer_text(pred_text: Any) -> str:
    text = '' if pred_text is None else str(pred_text)
    match = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r'\s+', ' ', match.group(1)).strip()
    return re.sub(r'\s+', ' ', text).strip()


def _compute_completeness(result: Dict[str, Any]) -> float:
    gt_count = max(int(result.get('total_gt_count', 0)), 0)
    matched_count = max(int(result.get('total_matched_count', 0)), 0)
    if gt_count == 0:
        return 0.0
    return float(matched_count) / float(gt_count)


def _compute_accuracy(result: Dict[str, Any]) -> float:
    pred_count = max(int(result.get('total_pred_count', 0)), 0)
    matched_count = max(int(result.get('total_matched_count', 0)), 0)
    if pred_count == 0:
        return 0.0
    return float(matched_count) / float(pred_count)


def _get_dim_weights() -> Dict[str, float]:
    return {
        'Camera Movement': 1.0,
        'Shot Size': 1.0,
        'Depth of Field': 1.0,
        'Camera Angle': 1.0,
        'Composition': 1.0,
        'Subject Orientation': 1.0,
    }


def _compute_aspect_completeness(result: Dict[str, Any]) -> float:
    by_dim = result.get('by_dim', {}) if isinstance(result, dict) else {}
    dim_weights = _get_dim_weights()
    weighted_sum = 0.0
    total_weight = 0.0
    for dim, weight in dim_weights.items():
        dim_value = by_dim.get(dim, {}) if isinstance(by_dim, dict) else {}
        gt_count = max(int(dim_value.get('gt_count', 0)), 0)
        matched_count = max(int(dim_value.get('matched_count', 0)), 0)
        score = float(matched_count) / float(gt_count) if gt_count > 0 else 0.0
        weighted_sum += weight * score
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def _compute_aspect_accuracy(result: Dict[str, Any]) -> float:
    by_dim = result.get('by_dim', {}) if isinstance(result, dict) else {}
    dim_weights = _get_dim_weights()
    weighted_sum = 0.0
    total_weight = 0.0
    for dim, weight in dim_weights.items():
        dim_value = by_dim.get(dim, {}) if isinstance(by_dim, dict) else {}
        pred_count = max(int(dim_value.get('pred_count', 0)), 0)
        matched_count = max(int(dim_value.get('matched_count', 0)), 0)
        score = float(matched_count) / float(pred_count) if pred_count > 0 else 0.0
        weighted_sum += weight * score
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def _compute_aspect_coverage(result: Dict[str, Any], acc_threshold: float = 0.75) -> float:
    accuracy = _compute_accuracy(result)
    if accuracy <= acc_threshold:
        return 0.0
    gt_count = max(int(result.get('total_gt_count', 0)), 0)
    pred_count = max(int(result.get('total_pred_count', 0)), 0)
    if gt_count == 0:
        return 0.0
    return -max(0.0, float(gt_count - pred_count) / float(gt_count))


def _compute_match_reward(result: Dict[str, Any], acc_threshold: float = 0.75) -> float:
    accuracy = _compute_accuracy(result)
    if accuracy <= acc_threshold:
        return 0.0
    gt_count = max(int(result.get('total_gt_count', 0)), 0)
    pred_count = max(int(result.get('total_pred_count', 0)), 0)
    if gt_count == 0:
        return 0.0
    return -min(1.0, abs(float(gt_count - pred_count)) / float(gt_count))


def _compute_f1_reward(result: Dict[str, Any]) -> float:
    completeness = _compute_completeness(result)
    accuracy = _compute_accuracy(result)
    if completeness <= 0.0 or accuracy <= 0.0:
        return 0.0
    return 2.0 * completeness * accuracy / (completeness + accuracy)


def _compute_mixed_match_reward(result: Dict[str, Any], acc_threshold: float = 0.75, offset: float = 0.2) -> float:
    accuracy = _compute_accuracy(result)
    if accuracy <= acc_threshold:
        return 0.0
    gt_count = max(int(result.get('total_gt_count', 0)), 0)
    pred_count = max(int(result.get('total_pred_count', 0)), 0)
    if gt_count == 0:
        return 0.0
    return offset - min(1.0, abs(float(gt_count - pred_count)) / float(gt_count))


def _token_count(text: str) -> int:
    text = re.sub(r'\s+', ' ', text or '').strip()
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _compute_token_length_reward(pred_len: int, min_answer_tokens: int, min_penalty_tokens: int) -> float:
    if pred_len <= 0:
        return -1.0
    if min_penalty_tokens >= min_answer_tokens:
        min_penalty_tokens = max(0, min_answer_tokens - 1)
    if pred_len >= min_answer_tokens:
        return 0.0
    if pred_len <= min_penalty_tokens:
        return -1.0
    span = max(1, min_answer_tokens - min_penalty_tokens)
    return -float(min_answer_tokens - pred_len) / float(span)


class CineCapCompletenessReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_completeness(result))
            except Exception as e:
                print(f'CineCapCompletenessReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapAccuracyReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_accuracy(result))
            except Exception as e:
                print(f'CineCapAccuracyReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapAspectCoverageReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()
        self.acc_threshold = float(os.getenv('CINECAP_ACC_THRESHOLD', '0.75'))

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_aspect_coverage(result, acc_threshold=self.acc_threshold))
            except Exception as e:
                print(f'CineCapAspectCoverageReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapAspCompletenessReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_aspect_completeness(result))
            except Exception as e:
                print(f'CineCapAspCompletenessReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapAspAccuracyReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_aspect_accuracy(result))
            except Exception as e:
                print(f'CineCapAspAccuracyReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapMatchReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()
        self.acc_threshold = float(os.getenv('CINECAP_ACC_THRESHOLD', '0.75'))

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_match_reward(result, acc_threshold=self.acc_threshold))
            except Exception as e:
                print(f'CineCapMatchReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapF1Reward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_f1_reward(result))
            except Exception as e:
                print(f'CineCapF1Reward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapMixedMatchReward(ORM):
    def __init__(self):
        self.judge = CineCapGeminiJudge()
        self.acc_threshold = float(os.getenv('CINECAP_ACC_THRESHOLD', '0.75'))

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                gt_text = _normalize_solution_item(gt_item)
                pred_text = _extract_answer_text(completion)
                result = self.judge.score_pair(gt_text=gt_text, pred_text=pred_text)
                rewards.append(_compute_mixed_match_reward(result, acc_threshold=self.acc_threshold))
            except Exception as e:
                print(f'CineCapMixedMatchReward failed: {e}')
                rewards.append(0.0)
        return rewards


class CineCapTokenLengthReward(ORM):
    def __init__(self):
        self.min_answer_tokens = int(os.getenv('CINECAP_MIN_ANSWER_TOKENS', '73'))
        self.min_penalty_tokens = int(os.getenv('CINECAP_MIN_PENALTY_TOKENS', '40'))

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, gt_item in zip(completions, solution):
            try:
                del gt_item
                pred_text = _extract_answer_text(completion)
                pred_len = _token_count(pred_text)
                rewards.append(
                    _compute_token_length_reward(
                        pred_len,
                        min_answer_tokens=self.min_answer_tokens,
                        min_penalty_tokens=self.min_penalty_tokens))
            except Exception as e:
                print(f'CineCapTokenLengthReward failed: {e}')
                rewards.append(0.0)
        return rewards


orms['external_cinecap_completeness_reward'] = CineCapCompletenessReward
orms['external_cinecap_accuracy_reward'] = CineCapAccuracyReward
orms['external_cinecap_aspect_coverage_reward'] = CineCapAspectCoverageReward
orms['external_cinecap_asp_cmp_reward'] = CineCapAspCompletenessReward
orms['external_cinecap_asp_acc_reward'] = CineCapAspAccuracyReward
orms['external_cinecap_match_reward'] = CineCapMatchReward
orms['external_cinecap_f1_reward'] = CineCapF1Reward
orms['external_cinecap_mixed_match_reward'] = CineCapMixedMatchReward
orms['external_cinecap_token_length_reward'] = CineCapTokenLengthReward
