"""
6가지 remasking 설정 비교 - MATH 벤치마크

사용법:
    cd /workspace/dllm_unceratinty/opencompass
    python run.py examples/llada_6configs_math.py
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_gen import math_datasets

from opencompass.models import LLaDAModel

datasets = math_datasets

MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'

NUM_GPUS = 1

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
    diff_logits_eos_inf=True,
)

models = [
    dict(**common_config, abbr='llada-8b-low_confidence', remasking='low_confidence', use_ensemble_llada=False),
    dict(**common_config, abbr='llada-8b-entropy', remasking='entropy', use_ensemble_llada=False),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b1.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=1.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a0.0-b1.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=0.0, beta=1.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b0.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=0.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b0.0-clean', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=0.0, mlp_dropout_p=0.1, sample_from_clean_logits=True),
]

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8, num_split=None, min_task_size=16),
    runner=dict(type=LocalRunner, max_num_workers=64, task=dict(type=OpenICLInferTask), retry=5),
)
