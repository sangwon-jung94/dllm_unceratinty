"""
MATH benchmark with uncertainty-aware LLaDA model.

This configuration runs MATH dataset with:
- EnsembleLLaDA for efficient uncertainty estimation
- Uncertainty-aware remasking strategy

Usage:
    cd /workspace/dllm_unceratinty/opencompass
    python run.py examples/llada_instruct_math_uncertainty.py
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_gen import math_datasets

from opencompass.models import LLaDAModel

datasets = math_datasets

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct-uncertainty-math',
        path='/mnt/oujingyang/assets/model/LLaDA',  # Update this path
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # Uncertainty-aware parameters
        use_ensemble_llada=True,
        remasking='uncertainty_aware',
        mc_samples=8,
        alpha=1.0,
        beta=1.0,
        mlp_dropout_p=0.1,
        # Generation parameters - longer for MATH
        gen_steps=512,
        gen_length=512,
        gen_blocksize=64,
        diff_confidence_eos_eot_inf=True,
        diff_logits_eos_inf=False,
    )
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
