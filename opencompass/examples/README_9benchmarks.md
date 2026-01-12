# OpenCompass로 LLaDA 9개 벤치마크 실행 가이드

## 지원 벤치마크
| 벤치마크 | 분류 | 설명 |
|---------|------|------|
| MMLU | 지식 | 57개 과목의 다지선다 문제 |
| MMLU-Pro | 지식 (고급) | 더 어려운 MMLU 버전 |
| HellaSwag | 상식 추론 | 문장 완성 |
| ARC-C | 과학 추론 | ARC Challenge 과학 문제 |
| GSM8K | 수학 | 초등 수학 word problem |
| MATH | 수학 (고급) | 경시대회 수준 수학 |
| GPQA | 과학 QA | 대학원 수준 과학 문제 |
| HumanEval | 코드 생성 | Python 함수 생성 |
| MBPP | 코드 생성 | Python 기초 프로그래밍 |

## 6가지 설정
1. **low_confidence**: 표준 confidence 기반 remasking
2. **entropy**: entropy 기반 remasking
3. **uncertainty_aware (α=1.0, β=1.0)**: epistemic + aleatoric 균형
4. **uncertainty_aware (α=0.0, β=1.0)**: aleatoric only
5. **uncertainty_aware (α=1.0, β=0.0)**: epistemic only
6. **uncertainty_aware + clean sampling**: epistemic only, clean logits에서 샘플링

## 실행 방법

### 1. 기본 실행 (단일 GPU)
```bash
cd /workspace/dllm_unceratinty/opencompass
python run.py examples/llada_6configs_9benchmarks.py
```

### 2. 멀티 GPU 병렬 실행 (4개 GPU 예시)
```bash
cd /workspace/dllm_unceratinty/opencompass

# CUDA_VISIBLE_DEVICES로 사용할 GPU 지정 후 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py examples/llada_6configs_9benchmarks.py
```

### 3. 특정 벤치마크만 실행
개별 벤치마크 설정 파일 사용:
```bash
# GSM8K만
python run.py examples/llada_6configs_gsm8k.py

# MATH만
python run.py examples/llada_6configs_math.py
```

### 4. 병렬화 옵션 조정
`llada_6configs_9benchmarks.py`의 `infer` 설정에서:
```python
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=4,  # 동시 실행할 태스크 수 = GPU 수
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,  # 최대 프로세스 수
        ...
    ),
)
```

### 5. Slurm 클러스터 사용
설정 파일의 주석 처리된 `SlurmRunner` 섹션 참조

## 결과 확인
결과는 `outputs/` 디렉토리에 저장됩니다:
- 각 (모델, 데이터셋) 조합별 결과
- 요약 테이블 자동 생성

## 메모리 최적화
- `memory_efficient=True` (기본값): GPU 메모리 절약 모드
- `batch_size=1`: 메모리 부족 시 유지
- `NUM_GPUS_PER_MODEL=2`: 큰 모델은 텐서 병렬 사용

## 주요 파일
- 설정: `opencompass/examples/llada_6configs_9benchmarks.py`
- 모델 래퍼: `opencompass/opencompass/models/dllm.py`
- 생성 코드: `generate.py`
