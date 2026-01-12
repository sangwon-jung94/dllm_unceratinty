# OpenCompass with Uncertainty-Aware LLaDA

이 문서는 OpenCompass에서 EnsembleLLaDA와 MC Dropout 기반 uncertainty-aware 생성을 사용하는 방법을 설명합니다.

## 개요

LLaDA 모델에 대해 다음 기능을 지원합니다:
- **EnsembleLLaDA**: 마지막 레이어에서 MLP Dropout 앙상블을 통한 효율적인 MC 샘플링
- **Uncertainty-aware remasking**: epistemic(인식론적) 및 aleatoric(우연적) 불확실성을 기반으로 한 리마스킹 전략
- **다양한 벤치마크 지원**: GSM8K, MATH, BBH, HumanEval, MBPP 등

## 설치

```bash
cd /workspace/dllm_unceratinty/opencompass
pip install -e .
```

## 기본 사용법

### 1. Uncertainty-aware GSM8K 실행

```bash
cd /workspace/dllm_unceratinty/opencompass
python run.py examples/llada_instruct_gsm8k_uncertainty.py
```

### 2. 여러 remasking 전략 비교

```bash
python run.py examples/llada_gsm8k_compare_remasking.py
```

## 주요 파라미터

### Uncertainty 관련 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| `use_ensemble_llada` | EnsembleLLaDA 사용 여부 (효율적인 MC 샘플링) | `False` |
| `remasking` | 리마스킹 전략 (`'low_confidence'`, `'random'`, `'uncertainty_aware'`) | `'low_confidence'` |
| `mc_samples` | MC Dropout 샘플 수 | `8` |
| `alpha` | Epistemic uncertainty 가중치 | `1.0` |
| `beta` | Aleatoric uncertainty 가중치 | `1.0` |
| `dropout_p` | MC Dropout 확률 (standard model) | `None` |
| `mlp_dropout_p` | MLP Dropout 확률 (EnsembleLLaDA) | `None` |

### 생성 관련 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| `gen_steps` | 생성 스텝 수 | `512` |
| `gen_length` | 생성 길이 | `512` |
| `gen_blocksize` | 블록 크기 (semi-autoregressive) | `512` |
| `temperature` | 샘플링 온도 | `0.0` |
| `cfg` | Classifier-free guidance 스케일 | `0` |

## 설정 예제

### 기본 Uncertainty-aware 설정

```python
from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct-uncertainty',
        path='/path/to/LLaDA',
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # Uncertainty 설정
        use_ensemble_llada=True,
        remasking='uncertainty_aware',
        mc_samples=8,
        alpha=1.0,
        beta=1.0,
        mlp_dropout_p=0.1,
        # 생성 설정
        gen_steps=256,
        gen_length=256,
        gen_blocksize=32,
        diff_confidence_eos_eot_inf=True,
    )
]
```

### 표준 MC Dropout 설정 (EnsembleLLaDA 없이)

```python
models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-mc-dropout',
        path='/path/to/LLaDA',
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # MC Dropout 설정 (EnsembleLLaDA 없이)
        use_ensemble_llada=False,
        remasking='uncertainty_aware',
        mc_samples=8,
        alpha=1.0,
        beta=1.0,
        dropout_p=0.1,  # 기존 dropout 레이어 사용
        # 생성 설정
        gen_steps=256,
        gen_length=256,
        gen_blocksize=32,
    )
]
```

## Uncertainty 해석

### Epistemic Uncertainty (인식론적 불확실성)
- 모델이 "모르는" 것에 대한 불확실성
- 더 많은 데이터로 줄일 수 있음
- `alpha` 값을 높이면 epistemic uncertainty가 높은 위치의 unmask를 지연

### Aleatoric Uncertainty (우연적 불확실성)
- 데이터 자체의 본질적인 노이즈/모호성
- 더 많은 데이터로도 줄일 수 없음
- `beta` 값을 높이면 aleatoric uncertainty가 높은 위치의 unmask를 지연

### 권장 설정

- **수학 문제 (GSM8K, MATH)**: `alpha=1.0, beta=1.0` 또는 `alpha=2.0, beta=1.0`
  - Epistemic uncertainty를 더 고려하여 확실한 추론 경로 선호
- **코드 생성 (HumanEval, MBPP)**: `alpha=1.0, beta=1.5`
  - 구문적 모호성(aleatoric)에 더 주의
- **추론 (BBH)**: `alpha=1.5, beta=1.0`
  - 논리적 확실성(epistemic)에 집중

## 예제 파일 위치

```
opencompass/examples/
├── llada_instruct_gsm8k_uncertainty.py      # GSM8K with uncertainty
├── llada_instruct_math_uncertainty.py       # MATH with uncertainty
├── llada_instruct_bbh_uncertainty.py        # BBH with uncertainty
└── llada_gsm8k_compare_remasking.py         # Compare remasking strategies
```

## 결과 확인

OpenCompass는 자동으로 결과를 수집하고 요약합니다:

```bash
# 결과 요약 보기
python tools/summary.py outputs/<run_name>/
```

## 참고

- `visualize_uncertainty_by_timestep.py`: 개별 샘플에 대한 상세한 uncertainty 분석
- `generate.py`: 핵심 생성 및 uncertainty 계산 로직
- `models/EnsembleLLaDA.py`: EnsembleLLaDA 구현
