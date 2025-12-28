# Experiment Scripts

이 디렉토리에는 다양한 uncertainty visualization 실험을 위한 스크립트들이 있습니다.

## 사용법

```bash
cd /workspace/dllm_unceratinty
./scripts/<script_name>.sh
```

## 사용 가능한 스크립트

### 1. Quick Test
```bash
./scripts/run_quick_test.sh
```
- 빠른 테스트용 (짧은 길이, 적은 스텝)
- Custom prompt 사용

### 2. No Prompt (Unconditional)
```bash
./scripts/run_no_prompt.sh
```
- 프롬프트 없이 무조건부 생성
- Uncertainty 변화 관찰용
- Base 모델 사용

### 3. GSM8K Benchmark - Full Dataset
```bash
./scripts/run_gsm8k.sh
```
- GSM8K 전체 데이터셋 (1319 samples)
- 배치 단위로 처리하여 메모리 효율적

### 4. GSM8K - Single Sample (Uncertainty Visualization)
```bash
./scripts/run_gsm8k_single.sh
```
- 단일 샘플로 uncertainty 분해 시각화
- MC Dropout 사용하여 epistemic/aleatoric 분리

### 5. MC Samples Comparison
```bash
./scripts/run_mc_comparison.sh
```
- 다양한 MC Dropout 샘플 수 비교 (4, 8, 16)

### 6. Alpha/Beta Comparison
```bash
./scripts/run_alpha_beta_comparison.sh
```
- 다양한 α, β 파라미터 조합 비교

### 7. Remasking Strategy Comparison
```bash
./scripts/run_remasking_comparison.sh
```
- Uncertainty-aware vs Low-confidence vs Random 비교

### 8. All Benchmarks
```bash
./scripts/run_all_benchmarks.sh
```
- 여러 벤치마크에서 실험 실행

## 주요 파라미터

- `--num_samples`: 벤치마크에서 처리할 샘플 수 (`None` = 전체 데이터셋)
- `--batch_size`: 한 번에 처리할 배치 크기 (메모리 제약 고려)
- `--remasking`: Remasking strategy (`uncertainty_aware`, `low_confidence`, `random`)
- `--mc_samples`: MC Dropout 샘플 수 (uncertainty_aware에서만 사용)
- `--alpha`: Epistemic uncertainty 가중치
- `--beta`: Aleatoric uncertainty 가중치
- `--device`: 사용할 GPU (예: `cuda:2`)
- `--exp_name`: 실험 이름 (결과 디렉토리명)
- `--logits_eos_inf`: EOS 토큰의 logits를 -inf로 설정 (조기 종료 방지)
- `--confidence_eos_eot_inf`: EOS/EoT 토큰의 confidence를 -inf로 설정

### 배치 처리

전체 벤치마크를 처리할 때는 배치 크기를 적절히 설정하여 메모리 문제를 방지합니다:

```bash
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples None \  # 전체 데이터셋
    --batch_size 4 \      # 한 번에 4개씩 처리
    --device cuda:2
```

GSM8K 전체: 1319 samples → 배치 크기 4 → 330 batches

### EOS 토큰 처리 (LLaDA 논문 Appendix B.4)

LLaDA 논문에 따르면, pure diffusion sampling에서 EOS 토큰 비율이 매우 높아져 생성이 짧아지는 문제가 있습니다. 
이를 완화하기 위해 HumanEval, MBPP, GSM8K, Math, GPQA 등의 벤치마크에서는:
- `--logits_eos_inf` 플래그를 사용하여 EOS 토큰의 confidence score를 0으로 설정
- 이를 통해 적절한 길이의 텍스트 생성 가능

## 결과 확인

모든 결과는 `./result/<exp_name>/` 디렉토리에 저장됩니다:
- `output.txt`: 생성된 텍스트
- `uncertainty_data.npz`: Uncertainty 데이터
- `uncertainty_over_timesteps.png`: Timestep별 uncertainty 그래프
- `uncertainty_components.png`: Uncertainty 분해 그래프
