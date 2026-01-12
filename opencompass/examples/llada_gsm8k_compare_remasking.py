"""
GSM8K benchmark comparing different remasking strategies.

This configuration allows comparison between:
1. low_confidence: Standard confidence-based remasking
2. random: Random remasking
3. uncertainty_aware: MC Dropout-based uncertainty remasking (epistemic + aleatoric)

Usage:
    cd /workspace/dllm_unceratinty/opencompass
    
    # Run all strategies:
    python run.py examples/llada_gsm8k_compare_remasking.py
    
    # Or run individual strategies by modifying 'models' list below
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

from opencompass.models import LLaDAModel

datasets = gsm8k_datasets

# Common configuration
base_config = dict(
    type=LLaDAModel,
    path='/mnt/oujingyang/assets/model/LLaDA',  # Update this path
    max_out_len=512,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    gen_steps=256,
    gen_length=256,
    gen_blocksize=32,
    diff_confidence_eos_eot_inf=True,
    diff_logits_eos_inf=False,
)

models = [
    # 1. Standard low_confidence remasking
    dict(
        **base_config,
        abbr='llada-8b-low_confidence',
        remasking='low_confidence',
        use_ensemble_llada=False,
    ),
    
    # 2. Random remasking
    dict(
        **base_config,
        abbr='llada-8b-random',
        remasking='random',
        use_ensemble_llada=False,
    ),
    
    # 3. Uncertainty-aware with EnsembleLLaDA (alpha=1.0, beta=1.0)
    dict(
        **base_config,
        abbr='llada-8b-uncertainty-a1b1',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=1.0,
        beta=1.0,
        mlp_dropout_p=0.1,
    ),
    
    # 4. Uncertainty-aware with higher epistemic weight (alpha=2.0, beta=1.0)
    dict(
        **base_config,
        abbr='llada-8b-uncertainty-a2b1',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=2.0,
        beta=1.0,
        mlp_dropout_p=0.1,
    ),
    
    # 5. Uncertainty-aware with higher aleatoric weight (alpha=1.0, beta=2.0)
    dict(
        **base_config,
        abbr='llada-8b-uncertainty-a1b2',
        remasking='uncertainty_aware',
        use_ensemble_llada=True,
        mc_samples=8,
        alpha=1.0,
        beta=2.0,
        mlp_dropout_p=0.1,
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
