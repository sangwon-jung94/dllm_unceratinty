"""
6가지 remasking 설정 비교 - 9개 주요 벤치마크 통합 실행

벤치마크:
- MMLU (지식)
- MMLU-Pro (지식, 고급)
- HellaSwag (상식 추론)
- ARC-C (과학 추론)
- GSM8K (수학)
- MATH (수학, 고급)
- GPQA (과학 QA)
- HumanEval (코드 생성)
- MBPP (코드 생성)

설정:
1. low_confidence: 표준 confidence 기반 remasking
2. entropy: entropy 기반 remasking  
3. uncertainty_aware (α=1.0, β=1.0): epistemic + aleatoric 균형
4. uncertainty_aware (α=0.0, β=1.0): aleatoric only
5. uncertainty_aware (α=1.0, β=0.0): epistemic only
6. uncertainty_aware (α=1.0, β=0.0) + clean sampling: epistemic only, clean logits에서 샘플링

공통: gen_steps=256, gen_length=256, gen_blocksize=256, LLaDA-8B-Instruct

사용법:
    cd /workspace/dllm_unceratinty/opencompass
    
    # 기본 실행 (단일 GPU)
    python run.py examples/llada_6configs_9benchmarks.py
    
    # 여러 GPU 병렬 실행 (예: 4개 GPU에서 동시에 다른 모델/데이터셋 실행)
    python run.py examples/llada_6configs_9benchmarks.py --slurm -p <partition> --max-num-workers 4
    
    # 또는 로컬에서 여러 프로세스로 병렬 실행
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py examples/llada_6configs_9benchmarks.py
"""
from mmengine.config import read_base

with read_base():
    # 9개 벤치마크 import
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.math.math_gen import math_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets

from opencompass.models import LLaDAModel

# 모든 데이터셋 합치기
datasets = (
    mmlu_datasets +
    mmlu_pro_datasets +
    hellaswag_datasets +
    ARC_c_datasets +
    gsm8k_datasets + 
    math_datasets +
    gpqa_datasets +
    humaneval_datasets + 
    mbpp_datasets
)

MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'

# 멀티 GPU 설정
# - NUM_GPUS_PER_MODEL: 한 모델이 사용할 GPU 수 (큰 모델일 때 증가)
# - 실제 병렬화는 OpenCompass의 partitioner/runner가 처리
NUM_GPUS_PER_MODEL = 1

common_config = dict(
    type=LLaDAModel,
    path=MODEL_PATH,
    max_out_len=256,
    batch_size=1,
    run_cfg=dict(num_gpus=NUM_GPUS_PER_MODEL),
    gen_steps=256,
    gen_length=256,
    gen_blocksize=256,
    diff_confidence_eos_eot_inf=True,
    diff_logits_eos_inf=True,
    memory_efficient=True,  # GPU 메모리 절약
)

models = [
    # 1. low_confidence
    dict(
        **common_config,
        abbr='llada-8b-low_confidence',
        remasking='low_confidence',
        use_ensemble_llada=False,
    ),
    
    # 2. entropy
    dict(
        **common_config,
        abbr='llada-8b-entropy',
        remasking='entropy',
        use_ensemble_llada=False,
    ),
    
    # 3. uncertainty_aware (α=1.0, β=1.0)
    dict(
        **common_config,
        abbr='llada-8b-ua-a1.0-b1.0',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=1.0,
        beta=1.0,
        mlp_dropout_p=0.1,
    ),
    
    # 4. uncertainty_aware (α=0.0, β=1.0) - aleatoric only
    dict(
        **common_config,
        abbr='llada-8b-ua-a0.0-b1.0',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=0.0,
        beta=1.0,
        mlp_dropout_p=0.1,
    ),
    
    # 5. uncertainty_aware (α=1.0, β=0.0) - epistemic only
    dict(
        **common_config,
        abbr='llada-8b-ua-a1.0-b0.0',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=1.0,
        beta=0.0,
        mlp_dropout_p=0.1,
    ),
    
    # 6. uncertainty_aware (α=1.0, β=0.0) + clean sampling
    dict(
        **common_config,
        abbr='llada-8b-ua-a1.0-b0.0-clean',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=1.0,
        beta=0.0,
        mlp_dropout_p=0.1,
        sample_from_clean_logits=True,
    ),
]

# =============================================================================
# 병렬 실행 설정
# =============================================================================
# OpenCompass는 태스크 레벨 병렬화를 지원합니다:
# - 각 (모델, 데이터셋) 조합이 하나의 태스크
# - 여러 GPU에서 다른 태스크를 동시에 실행
# - 6개 모델 x 9개 벤치마크 = 54개 태스크를 병렬로 분배

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        # num_worker: 동시에 실행할 태스크(GPU) 수
        # 보유한 GPU 수에 맞게 설정 (예: 4개 GPU면 num_worker=4)
        num_worker=4,
        num_split=None,
        min_task_size=1,  # 작은 태스크도 분배
    ),
    runner=dict(
        type=LocalRunner,
        # max_num_workers: 최대 동시 프로세스 수
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
        retry=3
    ),
)

# =============================================================================
# Slurm 클러스터 사용 시 (선택사항)
# =============================================================================
# Slurm을 사용하려면 아래 주석을 해제하고 위의 infer를 주석 처리하세요:
#
# from opencompass.runners import SlurmRunner
# infer = dict(
#     partitioner=dict(
#         type=NumWorkerPartitioner,
#         num_worker=8,  # 동시에 실행할 GPU/노드 수
#     ),
#     runner=dict(
#         type=SlurmRunner,
#         partition='gpu',  # Slurm 파티션 이름
#         quotatype='reserved',  # 또는 'spot', 'auto'
#         max_num_workers=16,
#         task=dict(type=OpenICLInferTask),
#         retry=3
#     ),
# )
