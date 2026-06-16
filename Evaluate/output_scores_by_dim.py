import argparse
import json

DIM_KEYS = [
    'Camera Movement',
    'Shot Size',
    'Depth of Field',
    'Camera Angle',
    'Composition',
    'Subject Orientation',
]


def safe_ratio(num, den):
    if den is None or den <= 0:
        return 0.0
    return float(num) / float(den)


def calculate_average_scores(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_com_scores = []
    total_acc_scores = []
    dim_com_scores = {dim: [] for dim in DIM_KEYS}
    dim_acc_scores = {dim: [] for dim in DIM_KEYS}

    for item in data:
        if 'com' not in item or 'acc' not in item:
            continue
        try:
            com = float(item['com'])
            acc = float(item['acc'])
        except (ValueError, TypeError):
            continue

        # if acc == 0.0:
        #     continue

        checklist = item.get('checklist', {})
        by_dim = checklist.get('by_dim', {}) if isinstance(checklist, dict) else {}

        total_com_scores.append(com)
        total_acc_scores.append(acc)

        for dim in DIM_KEYS:
            dim_info = by_dim.get(dim, {}) if isinstance(by_dim, dict) else {}
            gt_count = int(dim_info.get('gt_count', 0) or 0)
            pred_count = int(dim_info.get('pred_count', 0) or 0)
            matched_count = int(dim_info.get('matched_count', 0) or 0)

            dim_com_scores[dim].append(safe_ratio(matched_count, gt_count))
            dim_acc_scores[dim].append(safe_ratio(matched_count, pred_count))

    total_samples = len(total_com_scores)
    if total_samples == 0:
        print('❌ 错误: 未在文件中找到任何有效的分数数据！')
        return

    avg_total_com = sum(total_com_scores) / total_samples
    avg_total_acc = sum(total_acc_scores) / total_samples

    print('-' * 60)
    print('📊 Final Evaluation Metrics')
    print('-' * 60)
    print(f'✅ Total Valid Samples: {total_samples}')
    print(f'📈 Avg Overall Comprehensiveness: {avg_total_com:.4f}')
    print(f'🎯 Avg Overall Accuracy:          {avg_total_acc:.4f}')
    print('-' * 60)
    print('📌 Per-Dimension Metrics')
    print('-' * 60)

    for dim in DIM_KEYS:
        avg_dim_com = sum(dim_com_scores[dim]) / total_samples if total_samples else 0.0
        avg_dim_acc = sum(dim_acc_scores[dim]) / total_samples if total_samples else 0.0
        print(f'{dim:<20} | cmp: {avg_dim_com:.4f} | acc: {avg_dim_acc:.4f}')

    print('-' * 60)


def main():
    parser = argparse.ArgumentParser(description='Calculate overall and per-dimension Comprehensiveness/Accuracy scores.')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='/m2v_intern/maoxinyu03/CineCap/evaluate/0225_visually_anchored_think_answer_infer_cinecap472/evaluate_0225_think_answer_gemini_output_by_dim.json'
    )
    args = parser.parse_args()
    calculate_average_scores(args.input)


if __name__ == '__main__':
    main()
