"""
GSM8K 결과 파일 채점 스크립트

output.txt 파일에서 모델의 답변을 추출하고, 
원본 GSM8K 데이터셋의 정답과 비교하여 정확도를 계산합니다.
"""

import re
import argparse
from pathlib import Path
from datasets import load_dataset


def extract_answer_from_boxed(text):
    """
    모델 답변에서 \boxed{답} 또는 \(\boxed{답}\) 형식의 답을 추출합니다.
    
    Args:
        text: 모델의 전체 답변 텍스트
        
    Returns:
        추출된 숫자 문자열 또는 None
    """
    # \boxed{답} 또는 \(\boxed{답}\) 형식 찾기
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{답}
        r'\\\(\\boxed\{([^}]+)\}\\\)',  # \(\boxed{답}\)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 마지막 boxed 답을 사용 (최종 답변)
            answer = matches[-1].strip()
            # 숫자만 추출 (쉼표, 달러 기호 등 제거)
            answer = answer.replace(',', '').replace('$', '').replace(' ', '')
            return answer
    
    return None


def extract_answer_from_ground_truth(answer_text):
    """
    GSM8K 정답에서 #### 뒤의 숫자를 추출합니다.
    
    Args:
        answer_text: GSM8K 원본 정답 텍스트
        
    Returns:
        정답 숫자 문자열
    """
    # #### 뒤의 숫자 추출
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        answer = match.group(1).strip()
        # 숫자만 추출 (쉼표 제거)
        answer = answer.replace(',', '').replace('$', '').replace(' ', '')
        return answer
    return None


def normalize_number(num_str):
    """
    숫자 문자열을 정규화합니다 (비교를 위해).
    
    Args:
        num_str: 숫자 문자열
        
    Returns:
        정규화된 숫자 문자열 또는 None
    """
    if num_str is None:
        return None
    
    try:
        # 정수로 변환 가능하면 정수로
        if '.' not in num_str:
            return str(int(num_str))
        else:
            # 소수인 경우 float로
            return str(float(num_str))
    except (ValueError, TypeError):
        # 숫자가 아닌 경우 원본 반환
        return num_str


def parse_output_file(output_file, max_samples=None):
    """
    output.txt 파일을 파싱하여 각 샘플의 질문과 답변을 추출합니다.
    
    Args:
        output_file: output.txt 파일 경로
        max_samples: 최대 샘플 수 (None이면 전체)
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    samples = []
    
    # [Sample N] 으로 구분되는 섹션 찾기
    sample_pattern = r'\[Sample (\d+)\]\s*Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\[Sample \d+\]|$)'
    matches = re.findall(sample_pattern, content, re.DOTALL)
    
    for sample_num, question, answer in matches:
        # 구분선 제거
        answer = re.sub(r'-{10,}', '', answer).strip()
        
        samples.append({
            'sample_num': int(sample_num),
            'question': question.strip(),
            'answer': answer.strip()
        })
        
        # max_samples 제한
        if max_samples is not None and len(samples) >= max_samples:
            break
    
    return samples


def evaluate_gsm8k(output_file, verbose=False, max_samples=None):
    """
    GSM8K 결과를 평가합니다.
    
    Args:
        output_file: output.txt 파일 경로
        verbose: 자세한 출력 여부
        max_samples: 최대 샘플 수 (None이면 전체)
        
    Returns:
        Dict with evaluation metrics
    """
    # output.txt 파싱
    print(f"Parsing output file: {output_file}")
    samples = parse_output_file(output_file, max_samples=max_samples)
    print(f"Found {len(samples)} samples" + (f" (limited to {max_samples})" if max_samples else ""))
    
    # GSM8K 데이터셋 로드
    print("Loading GSM8K dataset...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    
    # 평가
    correct = 0
    total = len(samples)
    results = []
    
    for i, sample in enumerate(samples):
        sample_idx = sample['sample_num'] - 1  # 0-based index
        
        if sample_idx >= len(dataset):
            print(f"Warning: Sample {sample['sample_num']} exceeds dataset size")
            continue
        
        # 정답 추출
        ground_truth_answer = extract_answer_from_ground_truth(dataset[sample_idx]['answer'])
        
        # 모델 답변 추출
        predicted_answer = extract_answer_from_boxed(sample['answer'])
        
        # 정규화
        ground_truth_norm = normalize_number(ground_truth_answer)
        predicted_norm = normalize_number(predicted_answer)
        
        # 비교
        is_correct = (ground_truth_norm == predicted_norm) and (ground_truth_norm is not None)
        
        if is_correct:
            correct += 1
        
        result = {
            'sample_num': sample['sample_num'],
            'question': sample['question'][:100] + '...',  # 처음 100자만
            'ground_truth': ground_truth_answer,
            'predicted': predicted_answer,
            'correct': is_correct
        }
        results.append(result)
        
        if verbose or not is_correct:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Sample {sample['sample_num']}:")
            print(f"  Question: {result['question']}")
            print(f"  Ground Truth: {ground_truth_answer}")
            print(f"  Predicted: {predicted_answer}")
            if not is_correct:
                print(f"  Normalized GT: {ground_truth_norm}")
                print(f"  Normalized Pred: {predicted_norm}")
    
    # 통계 계산
    accuracy = (correct / total * 100) if total > 0 else 0
    
    metrics = {
        'total_samples': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'results': results
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate GSM8K results from output.txt')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output.txt file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each sample')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save detailed results (JSON format)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    
    if not output_file.exists():
        print(f"Error: File not found: {output_file}")
        return
    
    # 평가 실행
    metrics = evaluate_gsm8k(output_file, verbose=args.verbose, max_samples=args.max_samples)
    
    # 결과 출력
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print("="*80)
    
    # 결과 저장
    if args.save_results:
        import json
        save_path = Path(args.save_results)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {save_path}")


if __name__ == '__main__':
    main()
