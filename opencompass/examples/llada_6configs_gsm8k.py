"""
6가지 remasking 설정 비교 - GSM8K 벤치마크

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
    python run.py examples/llada_6configs_gsm8k.py
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.models import LLaDAModel

# GSM8K 데이터셋 설정 (직접 정의)
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]

# 모델 경로 설정
MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'

# 멀티 GPU 설정
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
    memory_efficient=True,
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
