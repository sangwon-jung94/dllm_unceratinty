"""
6가지 remasking 설정 비교 - 그룹 A (MMLU + MATH)

포함된 벤치마크 (~19,000 샘플):
- MMLU (14,042) - 지식
- MATH (5,000) - 수학

사용법:
    cd /workspace/dllm_unceratinty/opencompass
    CUDA_VISIBLE_DEVICES=0,1 python run.py examples/llada_6configs_groupA.py
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MMLUDataset, MATHDataset, MATHEvaluator, AccEvaluator
from opencompass.models import LLaDAModel

# MMLU 설정
mmlu_reader_cfg = dict(input_columns=['input', 'A', 'B', 'C', 'D'], output_column='target')
mmlu_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
mmlu_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

mmlu_datasets = [dict(
    abbr='mmlu',
    type=MMLUDataset,
    path='opencompass/mmlu',
    reader_cfg=mmlu_reader_cfg,
    infer_cfg=mmlu_infer_cfg,
    eval_cfg=mmlu_eval_cfg)]

# MATH 설정
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Problem: {problem}\nSolution:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))
math_eval_cfg = dict(evaluator=dict(type=MATHEvaluator))

math_datasets = [dict(
    abbr='math',
    type=MATHDataset,
    path='opencompass/math',
    reader_cfg=math_reader_cfg,
    infer_cfg=math_infer_cfg,
    eval_cfg=math_eval_cfg)]

datasets = mmlu_datasets + math_datasets

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
    memory_efficient=True,
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
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4, num_split=None, min_task_size=16),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLInferTask), retry=5),
)
