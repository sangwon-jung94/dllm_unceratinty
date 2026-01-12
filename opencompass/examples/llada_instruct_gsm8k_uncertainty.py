"""
GSM8K benchmark with uncertainty-aware LLaDA model.

This configuration runs GSM8K with:
- EnsembleLLaDA for efficient uncertainty estimation
- Uncertainty-aware remasking strategy
- MC samples = 8, alpha = 1.0, beta = 1.0

Usage:
    cd /workspace/dllm_unceratinty/opencompass
    python run.py examples/llada_instruct_gsm8k_uncertainty.py
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

from opencompass.models import LLaDAModel

datasets = gsm8k_datasets

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct-uncertainty',
        path='/mnt/oujingyang/assets/model/LLaDA',  # Update this path to your LLaDA model
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # Uncertainty-aware parameters
        use_ensemble_llada=True,  # Use EnsembleLLaDA for efficient MC sampling
        remasking='uncertainty_aware',  # Use uncertainty-aware remasking
        mc_samples=8,  # Number of MC samples
        alpha=1.0,  # Epistemic uncertainty weight
        beta=1.0,  # Aleatoric uncertainty weight
        mlp_dropout_p=0.1,  # MLP dropout probability
        # Generation parameters
        gen_steps=256,
        gen_length=256,
        gen_blocksize=32,
        diff_confidence_eos_eot_inf=True,
        diff_logits_eos_inf=False,
    )
]

# Runner configuration
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
