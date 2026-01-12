"""
6가지 remasking 설정 비교 - 모든 벤치마크 통합 실행

포함된 벤치마크:
- GSM8K (수학)
- MATH (수학)
- BBH (추론)
- HumanEval (코드)
- MBPP (코드)
- MMLU (지식)
- HellaSwag (상식)

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
    python run.py examples/llada_6configs_all_benchmarks.py
    
    # 특정 벤치마크만 실행하려면 개별 파일 사용:
    python run.py examples/llada_6configs_gsm8k.py
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.math.math_gen import math_datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets

from opencompass.models import LLaDAModel

# 모든 데이터셋 합치기
datasets = (
    gsm8k_datasets + 
    # math_datasets + 
    bbh_datasets + 
    humaneval_datasets + 
    mbpp_datasets +
    # mmlu_datasets +
    hellaswag_datasets
)

MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'

# 멀티 GPU 설정
NUM_GPUS = 4  # 사용할 GPU 수 (1, 2, 4, 8 등)

common_config = dict(
    type=LLaDAModel,
    path=MODEL_PATH,
    max_out_len=256,
    batch_size=1,
    run_cfg=dict(num_gpus=NUM_GPUS),
    gen_steps=256,
    gen_length=256,
    gen_blocksize=256,
    diff_confidence_eos_eot_inf=True,
    diff_logits_eos_inf=True,  # EOS 토큰 logits -inf 설정
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

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
        num_split=None,
        min_task_size=16,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
