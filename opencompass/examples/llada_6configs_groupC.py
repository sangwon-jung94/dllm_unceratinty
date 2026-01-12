"""
6가지 remasking 설정 비교 - 그룹 C (ARC + PIQA + WinoGrande + BoolQ + CommonsenseQA + TruthfulQA)

포함된 벤치마크 (~9,600 샘플):
- ARC-Challenge (1,172) - 과학 추론
- PIQA (1,838) - 물리 상식
- WinoGrande (1,267) - 상식 추론
- BoolQ (3,270) - 독해 및 사실 검증
- CommonsenseQA (1,221) - 상식
- TruthfulQA (817) - 진실성 평가

사용법:
    cd /workspace/dllm_unceratinty/opencompass
    CUDA_VISIBLE_DEVICES=4,5 python run.py examples/llada_6configs_groupC.py
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import (
    ARCDataset,
    piqa_postprocess, PIQADataset,
    WinograndeDataset,
    BoolQDataset,
    commonsenseqa_postprocess, CommonsenseQADataset,
    TruthfulQADataset, TruthfulQAEvaluator
)
from opencompass.models import LLaDAModel

# ARC-Challenge 설정
arc_reader_cfg = dict(input_columns=['question', 'textA', 'textB', 'textC', 'textD'], output_column='answerKey')
arc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
arc_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

arc_datasets = [dict(
    abbr='arc_challenge',
    type=ARCDataset,
    path='opencompass/arc',
    name='ARC-Challenge',
    reader_cfg=arc_reader_cfg,
    infer_cfg=arc_infer_cfg,
    eval_cfg=arc_eval_cfg)]

# PIQA 설정
piqa_reader_cfg = dict(input_columns=['goal', 'sol1', 'sol2'], output_column='label')
piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Goal: {goal}\nA. {sol1}\nB. {sol2}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
piqa_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=piqa_postprocess, options='AB'))

piqa_datasets = [dict(
    abbr='piqa',
    type=PIQADataset,
    path='opencompass/piqa',
    reader_cfg=piqa_reader_cfg,
    infer_cfg=piqa_infer_cfg,
    eval_cfg=piqa_eval_cfg)]

# WinoGrande 설정
winogrande_reader_cfg = dict(input_columns=['opt1', 'opt2'], output_column='answer')
winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{sentence}\nA. {opt1}\nB. {opt2}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
winogrande_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'))

winogrande_datasets = [dict(
    abbr='winogrande',
    type=WinograndeDataset,
    path='opencompass/winogrande',
    reader_cfg=winogrande_reader_cfg,
    infer_cfg=winogrande_infer_cfg,
    eval_cfg=winogrande_eval_cfg)]

# BoolQ 설정
boolq_reader_cfg = dict(input_columns=['passage', 'question'], output_column='answer')
boolq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Passage: {passage}\nQuestion: {question}\nAnswer (True or False):'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
boolq_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

boolq_datasets = [dict(
    abbr='boolq',
    type=BoolQDataset,
    path='opencompass/boolq',
    reader_cfg=boolq_reader_cfg,
    infer_cfg=boolq_infer_cfg,
    eval_cfg=boolq_eval_cfg)]

# CommonsenseQA 설정
commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey')
commonsenseqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=commonsenseqa_postprocess))

commonsenseqa_datasets = [dict(
    abbr='commonsenseqa',
    type=CommonsenseQADataset,
    path='opencompass/commonsenseqa',
    reader_cfg=commonsenseqa_reader_cfg,
    infer_cfg=commonsenseqa_infer_cfg,
    eval_cfg=commonsenseqa_eval_cfg)]

# TruthfulQA 설정
truthfulqa_reader_cfg = dict(input_columns=['question'], output_column='best_answer')
truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='Question: {question}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128))
truthfulqa_eval_cfg = dict(evaluator=dict(type=TruthfulQAEvaluator))

truthfulqa_datasets = [dict(
    abbr='truthfulqa',
    type=TruthfulQADataset,
    path='opencompass/truthfulqa',
    reader_cfg=truthfulqa_reader_cfg,
    infer_cfg=truthfulqa_infer_cfg,
    eval_cfg=truthfulqa_eval_cfg)]

datasets = arc_datasets + piqa_datasets + winogrande_datasets + boolq_datasets + commonsenseqa_datasets + truthfulqa_datasets

MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'
NUM_GPUS = 1

# Data parallel: 각 worker가 서로 다른 GPU에서 독립적으로 실행
# device_map='cuda:0' - OpenCompass가 각 worker에 CUDA_VISIBLE_DEVICES=N 설정하므로
# cuda:0이 해당 worker에 할당된 GPU를 가리킴
common_config = dict(
    type=LLaDAModel,
    path=MODEL_PATH,
    max_out_len=256,
    batch_size=1,
    run_cfg=dict(num_gpus=NUM_GPUS),
    model_kwargs=dict(device_map='cuda:0', torch_dtype='torch.bfloat16'),
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

# num_worker=4: 4개 GPU 사용 (CUDA_VISIBLE_DEVICES=4,5,6,7이면 각각 GPU 사용)
# max_num_workers=4: 동시에 최대 4개 작업 실행
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4, num_split=None, min_task_size=16),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLInferTask), retry=5),
)
