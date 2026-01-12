"""
LLaDA Instruct 8B model configuration with uncertainty-aware generation.

This configuration enables:
- EnsembleLLaDA for efficient MC sampling (single forward pass)
- Uncertainty-aware remasking strategy
- Epistemic and aleatoric uncertainty decomposition
"""
from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct-uncertainty',
        path='/mnt/oujingyang/assets/model/LLaDA',  # Update this path
        max_out_len=1024,
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
